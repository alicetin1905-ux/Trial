"""A deterministic stand-in for Jev, for tests and dry runs. Its answers are crude functions of the
snapshot. It is NOT a model of Jev's skill, and results produced with it say nothing about Jev."""

from __future__ import annotations

import math

from jevbot.jev.client import JevResult


def _softmax(d: dict[str, float]) -> dict[str, float]:
    m = max(d.values())
    e = {k: math.exp(v - m) for k, v in d.items()}
    s = sum(e.values())
    return {k: round(v / s, 6) for k, v in e.items()}


def _choice(probs: dict[str, float]) -> dict:
    top = max(probs, key=probs.get)
    return {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs}


class FakeJev:
    model = "fake-jev"

    async def ask(self, snapshot: dict[str, float], side: int) -> JevResult:
        g = snapshot.get
        trend = g("ema50_slope_1h_atr", 0.0) + 0.5 * g("dist_ema50_4h_atr", 0.0)
        vol = g("rv_ratio_4h_3d", 1.0)
        regime = _softmax(
            {
                "trending_up": trend,
                "trending_down": -trend,
                "ranging": 0.5 - abs(trend),
                "high_vol_chop": vol - 1.2,
            }
        )
        mom = g("ret_4h_z", 0.0) + g("flow_imb_1h", 0.0)
        direction = _softmax({"up": mom, "down": -mom, "flat": 0.3})
        pressure = 1 / (1 + math.exp(-4 * (g("flow_imb_15m", 0.0) + g("flow_imb_1h", 0.0))))
        q = max(0.0, min(4.0, 2.0 + side * trend - abs(g("pullback_6h_atr", 1.5) - 1.75)))
        lo = int(q)
        probs = {str(i): 0.0 for i in range(5)}
        probs[str(lo)] = round(1 - (q - lo), 6)
        if lo < 4:
            probs[str(lo + 1)] = round(q - lo, 6)
        risk = _softmax({"normal": 1.0, "elevated": vol - 1.0, "extreme": 2 * (vol - 1.6)})
        answers = {
            "regime": _choice(regime),
            "direction": _choice(direction),
            "buy_pressure_real": {"type": "noul", "noul": round(pressure, 6)},
            "setup_quality": {
                "type": "score",
                "score": round(q, 6),
                "confidence": 0.5,
                "probabilities": probs,
            },
            "risk_state": _choice(risk),
        }
        return JevResult(answers, self.model, 0.0, 0, 0.0, True)
