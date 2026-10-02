"""Jev client wrapper (architecture §5).

live profile:     timeout 1.5 s, SDK retries OFF, no cache. Any problem -> JevFailure -> no trade.
backtest profile: retries on, concurrency-capped, answers cached on disk by
                  sha256(state + questions + model alias), so reruns are free and repeatable.

A response from a model other than `pinned_model` comes back as an untrusted result: it is
logged, but decide() must not trade on it (architecture §5, /ackmodel).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path

from typesafe_sdk import (
    AsyncTypeSafeClient,
    RetryPolicy,
    TypeSafeAPIError,
    TypeSafeAPITimeoutError,
    TypeSafeAuthenticationError,
    TypeSafeError,
)

from jevbot.jev.questions import QUESTION_NAMES, Questions, build_state

JEV_USD_PER_INPUT_TOKEN = 0.042 / 1_000_000  # output tokens are free (published pricing, Sept 2026)
PROB_SUM_TOL = 0.02


@dataclass
class JevResult:
    answers: dict
    model: str
    latency_ms: float
    input_tokens: int
    cost_usd: float
    trusted: bool
    untrusted_reason: str | None = None
    cached: bool = False
    state: dict = field(default_factory=dict)


@dataclass
class JevFailure:
    reason: str  # timeout | error | auth | malformed
    detail: str
    latency_ms: float


class RateLimiter:
    """Spaces request starts at least 1/rate seconds apart (Jev: 1,200 requests/min)."""

    def __init__(self, rate_per_s: float) -> None:
        self.interval = 1.0 / rate_per_s
        self._next = 0.0
        self._lock = asyncio.Lock()

    async def wait(self) -> None:
        async with self._lock:
            now = time.perf_counter()
            start = max(now, self._next)
            self._next = start + self.interval
        await asyncio.sleep(max(0.0, start - now))


class JevCache:
    def __init__(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(str(path))
        self.db.execute("CREATE TABLE IF NOT EXISTS jev (k TEXT PRIMARY KEY, model TEXT, payload TEXT)")
        self.db.commit()

    @staticmethod
    def key(state: dict, wire: dict, model_alias: str) -> str:
        blob = json.dumps({"s": state, "q": wire, "m": model_alias}, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, k: str) -> tuple[str, dict] | None:
        row = self.db.execute("SELECT model, payload FROM jev WHERE k = ?", (k,)).fetchone()
        return None if row is None else (row[0], json.loads(row[1]))

    def put(self, k: str, model: str, payload: dict) -> None:
        self.db.execute("INSERT OR REPLACE INTO jev VALUES (?, ?, ?)", (k, model, json.dumps(payload)))
        self.db.commit()


def validate_answers(answers: dict) -> str | None:
    """Return a reason string if the answers are unusable, else None."""
    for name in QUESTION_NAMES:
        a = answers.get(name)
        if not isinstance(a, dict):
            return f"missing answer {name}"
        if a.get("type") == "noul":
            v = a.get("noul")
            if not isinstance(v, int | float) or not 0.0 <= v <= 1.0:
                return f"{name}: bad noul {v}"
            continue
        probs = a.get("probabilities") or {}
        vals = list(probs.values())
        if not vals or any(not 0.0 <= v <= 1.0 for v in vals) or abs(sum(vals) - 1.0) > PROB_SUM_TOL:
            return f"{name}: probabilities invalid {probs}"
        if a.get("type") == "score" and not (
            isinstance(a.get("score"), int | float) and math.isfinite(a["score"])
        ):
            return f"{name}: bad score"
    return None


class JevClient:
    def __init__(
        self,
        questions: Questions,
        profile: str,
        model: str,
        pinned_model: str,
        api_key: str | None = None,
        transport=None,
        live_timeout_s: float = 1.5,
        backtest_timeout_s: float = 30.0,
        cache: JevCache | None = None,
    ) -> None:
        if profile not in ("live", "backtest"):
            raise ValueError(profile)
        self.q, self.profile, self.model, self.pinned = questions, profile, model, pinned_model
        self.cache = cache if profile == "backtest" else None
        self.timeout = live_timeout_s if profile == "live" else backtest_timeout_s
        retry = (
            RetryPolicy(max_retries=0)
            if profile == "live"
            else RetryPolicy(max_retries=5, backoff_initial=1.0, backoff_max=30.0, timeout=180.0)
        )
        self._wire = questions.wire()
        self._sdk = AsyncTypeSafeClient(
            api_key=api_key, model=model, retry=retry, timeout=self.timeout, transport=transport
        )

    def _trust(self, model: str) -> tuple[bool, str | None]:
        if not self.pinned:
            return False, "model_not_pinned"
        if model != self.pinned:
            return False, "model_mismatch"
        return True, None

    async def ask(
        self, snapshot: dict[str, float], side: int, limiter: RateLimiter | None = None
    ) -> JevResult | JevFailure:
        state = build_state(snapshot, side, self.q.legend)
        key = JevCache.key(state, self._wire, self.model) if self.cache else None
        if self.cache and key:
            hit = self.cache.get(key)
            if hit is not None and (not self.pinned or hit[0] == self.pinned):
                model, payload = hit
                trusted, why = self._trust(model)
                return JevResult(
                    payload["answers"], model, 0.0, payload["input_tokens"], 0.0, trusted, why, True, state
                )

        if limiter is not None:  # only network calls spend rate budget; cache hits return above
            await limiter.wait()
        t0 = time.perf_counter()
        try:
            # Hard wall-clock guard on top of the SDK timeout: a slow answer is a missed answer.
            resp = await asyncio.wait_for(
                self._sdk.system_one(state=state, questions=self.q.questions),
                timeout=self.timeout + (0.1 if self.profile == "live" else 60.0),
            )
        except (TimeoutError, TypeSafeAPITimeoutError) as e:
            return JevFailure("timeout", type(e).__name__, (time.perf_counter() - t0) * 1000)
        except TypeSafeAuthenticationError as e:
            return JevFailure("auth", str(e)[:200], (time.perf_counter() - t0) * 1000)
        except (TypeSafeAPIError, TypeSafeError) as e:
            return JevFailure(
                "error", f"{type(e).__name__}: {str(e)[:200]}", (time.perf_counter() - t0) * 1000
            )
        latency = (time.perf_counter() - t0) * 1000

        body = resp.model_dump(mode="json")
        answers = body["answers"]
        bad = validate_answers(answers)
        if bad:
            return JevFailure("malformed", bad, latency)
        tokens = int(body["usage"]["input_tokens"])
        if self.cache and key:
            self.cache.put(key, body["model"], {"answers": answers, "input_tokens": tokens})
        trusted, why = self._trust(body["model"])
        return JevResult(
            answers,
            body["model"],
            latency,
            tokens,
            tokens * JEV_USD_PER_INPUT_TOKEN,
            trusted,
            why,
            False,
            state,
        )

    async def ask_many(
        self, items: list[tuple[dict, int]], concurrency: int, rate_per_s: float = 18.0
    ) -> list[JevResult | JevFailure]:
        """Backtest batch. Default 18 requests/s keeps under Jev's 1,200/min with headroom."""
        sem = asyncio.Semaphore(concurrency)
        limiter = RateLimiter(rate_per_s)

        async def one(snap, side):
            async with sem:
                return await self.ask(snap, side, limiter)

        return list(await asyncio.gather(*(one(s, d) for s, d in items)))

    async def aclose(self) -> None:
        await self._sdk.aclose()
