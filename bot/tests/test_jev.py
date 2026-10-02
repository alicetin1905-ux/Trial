import asyncio
import json
from pathlib import Path

import httpx2
import pytest

from jevbot.jev.client import JEV_USD_PER_INPUT_TOKEN, JevCache, JevClient, JevFailure, JevResult
from jevbot.jev.fake import FakeJev
from jevbot.jev.questions import QUESTION_NAMES, build_state, load_questions

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS = ROOT / "active" / "questions.yaml"
SNAP = {
    "ret_15m_z": 0.1,
    "ema50_slope_1h_atr": 0.4,
    "dist_ema50_4h_atr": 1.2,
    "pullback_6h_atr1h": 1.6,
    "rsi14_15m": 31.0,
    "flow_imb_15m": 0.2,
    "flow_imb_1h": 0.1,
    "rv_ratio_4h_3d": 0.9,
    "trade_count_z_15m": 0.3,
    "adx14_1h": 28.0,
}


def _answers(regime="trending_up"):
    return {
        "regime": {
            "type": "choice",
            "choice": regime,
            "confidence": 0.8,
            "probabilities": {
                "trending_up": 0.7,
                "trending_down": 0.05,
                "ranging": 0.2,
                "high_vol_chop": 0.05,
            },
        },
        "direction": {
            "type": "choice",
            "choice": "up",
            "confidence": 0.6,
            "probabilities": {"up": 0.55, "down": 0.15, "flat": 0.3},
        },
        "buy_pressure_real": {"type": "noul", "noul": 0.64},
        "setup_quality": {
            "type": "score",
            "score": 2.9,
            "confidence": 0.7,
            "legend": {"0": "a", "1": "b", "2": "c", "3": "d", "4": "e"},
            "probabilities": {"0": 0.0, "1": 0.05, "2": 0.2, "3": 0.55, "4": 0.2},
        },
        "risk_state": {
            "type": "choice",
            "choice": "normal",
            "confidence": 0.9,
            "probabilities": {"normal": 0.85, "elevated": 0.13, "extreme": 0.02},
        },
    }


class Server:
    """Fake TypeSafe API behind an httpx2 MockTransport."""

    def __init__(self, model="jev-1.13.0", status=200, delay=0.0, answers=None):
        self.model, self.status, self.delay = model, status, delay
        self.answers = answers if answers is not None else _answers()
        self.requests: list[dict] = []

    async def handler(self, request: httpx2.Request) -> httpx2.Response:
        self.requests.append(json.loads(request.content))
        if self.delay:
            await asyncio.sleep(self.delay)
        if self.status != 200:
            return httpx2.Response(self.status, json={"detail": "boom"})
        body = {
            "model": self.model,
            "answers": self.answers,
            "usage": {"input_tokens": 1000, "output_tokens": 30},
        }
        return httpx2.Response(200, json=body)

    def transport(self):
        return httpx2.MockTransport(self.handler)


def _client(server, profile="live", pinned="jev-1.13.0", cache=None):
    return JevClient(
        questions=load_questions(QUESTIONS),
        profile=profile,
        model="jev-latest",
        pinned_model=pinned,
        api_key="test-key-123456",
        transport=server.transport(),
        live_timeout_s=0.2,
        cache=cache,
    )


def run(coro):
    return asyncio.run(coro)


# --- questions & state -----------------------------------------------------------------------


def test_questions_file_has_the_five_spec_questions():
    q = load_questions(QUESTIONS)
    assert (
        set(q.questions)
        == set(QUESTION_NAMES)
        == {
            "regime",
            "direction",
            "buy_pressure_real",
            "setup_quality",
            "risk_state",
        }
    )
    wire = {k: v.model_dump() for k, v in q.questions.items()}
    assert wire["regime"]["type"] == "choice" and wire["setup_quality"]["type"] == "score"
    assert len(wire["setup_quality"]["criteria"]) == 5
    assert wire["buy_pressure_real"]["type"] == "noul"


def test_state_is_compact_and_stable():
    q = load_questions(QUESTIONS)
    s1 = build_state(SNAP, side=1, legend=q.legend)
    s2 = build_state(dict(reversed(list(SNAP.items()))), side=1, legend=q.legend)
    assert json.dumps(s1, sort_keys=True) == json.dumps(s2, sort_keys=True)
    assert s1["candidate_side"] == "long" and build_state(SNAP, 0, q.legend)["candidate_side"] == "none"
    assert s1["features"] == SNAP


# --- live profile ----------------------------------------------------------------------------


def test_live_call_returns_trusted_result_with_cost_and_latency():
    srv = Server()
    r = run(_client(srv).ask(SNAP, side=1))
    assert isinstance(r, JevResult) and r.trusted
    assert r.answers["regime"]["probabilities"]["trending_up"] == 0.7
    assert r.model == "jev-1.13.0"
    assert r.cost_usd == pytest.approx(1000 * JEV_USD_PER_INPUT_TOKEN)
    assert r.latency_ms >= 0
    assert srv.requests[0]["model"] == "jev-latest"
    assert set(srv.requests[0]["questions"]) == set(QUESTION_NAMES)


def test_model_change_makes_result_untrusted():
    r = run(_client(Server(model="jev-1.14.0")).ask(SNAP, side=1))
    assert isinstance(r, JevResult) and not r.trusted and r.untrusted_reason == "model_mismatch"


def test_unpinned_model_is_untrusted():
    r = run(_client(Server(), pinned="").ask(SNAP, side=1))
    assert not r.trusted and r.untrusted_reason == "model_not_pinned"


def test_live_timeout_is_a_failure_and_never_retried():
    srv = Server(delay=1.0)
    r = run(_client(srv).ask(SNAP, side=1))
    assert isinstance(r, JevFailure) and r.reason == "timeout"
    assert len(srv.requests) == 1


def test_live_server_error_is_a_failure_and_never_retried():
    srv = Server(status=503)
    r = run(_client(srv).ask(SNAP, side=1))
    assert isinstance(r, JevFailure) and r.reason == "error"
    assert len(srv.requests) == 1


def test_missing_answer_is_malformed():
    a = _answers()
    del a["risk_state"]
    r = run(_client(Server(answers=a)).ask(SNAP, side=1))
    assert isinstance(r, JevFailure) and r.reason == "malformed"


def test_probabilities_that_do_not_sum_to_one_are_malformed():
    a = _answers()
    a["regime"]["probabilities"] = {
        "trending_up": 0.9,
        "trending_down": 0.9,
        "ranging": 0.0,
        "high_vol_chop": 0.0,
    }
    r = run(_client(Server(answers=a)).ask(SNAP, side=1))
    assert isinstance(r, JevFailure) and r.reason == "malformed"


# --- backtest profile + cache ----------------------------------------------------------------


def test_backtest_cache_hit_needs_no_network(tmp_path):
    cache = JevCache(tmp_path / "jev_cache.sqlite")
    srv = Server()
    r1 = run(_client(srv, profile="backtest", cache=cache).ask(SNAP, side=1))
    srv2 = Server(status=500)  # would fail if called
    r2 = run(
        _client(srv2, profile="backtest", cache=JevCache(tmp_path / "jev_cache.sqlite")).ask(SNAP, side=1)
    )
    assert r2.answers == r1.answers and r2.cached and not srv2.requests


def test_cache_key_changes_with_questions_or_side(tmp_path):
    q = load_questions(QUESTIONS)
    k1 = JevCache.key(build_state(SNAP, 1, q.legend), q.wire(), "jev-latest")
    k2 = JevCache.key(build_state(SNAP, -1, q.legend), q.wire(), "jev-latest")
    wire = q.wire()
    wire["regime"]["instructions"] = "changed"
    k3 = JevCache.key(build_state(SNAP, 1, q.legend), wire, "jev-latest")
    assert len({k1, k2, k3}) == 3


def test_cached_answer_from_another_model_is_not_reused(tmp_path):
    cache = JevCache(tmp_path / "c.sqlite")
    run(
        _client(Server(model="jev-1.12.0"), profile="backtest", pinned="jev-1.12.0", cache=cache).ask(SNAP, 1)
    )
    srv = Server(model="jev-1.13.0")
    r = run(_client(srv, profile="backtest", pinned="jev-1.13.0", cache=cache).ask(SNAP, 1))
    assert r.model == "jev-1.13.0" and len(srv.requests) == 1


def test_ask_many_respects_concurrency_and_keeps_order(tmp_path):
    srv = Server()
    c = _client(srv, profile="backtest", cache=JevCache(tmp_path / "c.sqlite"))
    snaps = [SNAP | {"ret_15m_z": i / 10} for i in range(12)]
    out = run(c.ask_many([(s, 1) for s in snaps], concurrency=3))
    assert len(out) == 12 and all(isinstance(r, JevResult) for r in out)
    assert [json.loads(json.dumps(r.state["features"]))["ret_15m_z"] for r in out] == [
        i / 10 for i in range(12)
    ]


# --- fake ------------------------------------------------------------------------------------


def test_fake_jev_is_deterministic_and_well_formed():
    f = FakeJev()
    r1 = run(f.ask(SNAP, side=1))
    r2 = run(f.ask(SNAP, side=1))
    assert r1.answers == r2.answers and r1.trusted
    assert set(r1.answers) == set(QUESTION_NAMES)
    for a in r1.answers.values():
        if "probabilities" in a:
            assert sum(a["probabilities"].values()) == pytest.approx(1.0)


def test_client_closes_cleanly():
    async def go():
        c = _client(Server())
        await c.ask(SNAP, side=1)
        await c.aclose()

    run(go())


def test_ask_many_is_rate_limited(tmp_path):
    import time as _t

    c = _client(Server(), profile="backtest", cache=JevCache(tmp_path / "c.sqlite"))
    snaps = [(SNAP | {"ret_15m_z": i / 100}, 1) for i in range(11)]
    t0 = _t.perf_counter()
    run(c.ask_many(snaps, concurrency=8, rate_per_s=50.0))
    assert _t.perf_counter() - t0 >= 10 / 50.0 * 0.95  # 11 starts need >= 10 intervals of 20 ms


def test_cache_hits_do_not_consume_rate_budget(tmp_path):
    import time as _t

    cache = JevCache(tmp_path / "c.sqlite")
    c = _client(Server(), profile="backtest", cache=cache)
    snaps = [(SNAP | {"ret_15m_z": i / 100}, 1) for i in range(11)]
    run(c.ask_many(snaps, concurrency=8, rate_per_s=50.0))
    t0 = _t.perf_counter()
    run(c.ask_many(snaps, concurrency=8, rate_per_s=5.0))
    assert _t.perf_counter() - t0 < 0.5
