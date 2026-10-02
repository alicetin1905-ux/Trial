"""Strategy parameters, read from the YAML front matter of strategy.md.

strategy.md is what Opus may edit (in staging/). Unknown keys are rejected, so a candidate cannot
smuggle in anything the rules don't understand, and risk keys don't exist here at all: they live
only in config/risk.yaml.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from jevbot.strategy.combine import JevParams


class ParamError(ValueError):
    pass


class RuleParams(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pullback_min_atr: float = Field(gt=0)
    pullback_max_atr: float = Field(gt=0)
    rsi_long_max: float = Field(gt=0, lt=50)  # shorts use 100 - rsi_long_max
    stop_atr_k: float = Field(gt=0, le=5)
    take_profit_r: float = Field(gt=0, le=10)
    time_stop_bars: int = Field(gt=0, le=4 * 96)
    direction: Literal["both", "long", "short"]

    @model_validator(mode="after")
    def _ordered(self) -> RuleParams:
        if self.pullback_min_atr >= self.pullback_max_atr:
            raise ValueError("pullback_min_atr must be < pullback_max_atr")
        return self

    @classmethod
    def checked(cls, **kw) -> RuleParams:
        try:
            return cls(**kw)
        except ValidationError as e:
            raise ParamError(str(e)) from None


class StrategyFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    rules: RuleParams
    jev: JevParams | None = None  # absent = rules only (the M4 baseline)


@dataclass(frozen=True)
class Strategy:
    rules: RuleParams
    jev: JevParams | None
    body: str
    sha256: str


def load_strategy_md(path: Path) -> Strategy:
    raw = Path(path).read_text()
    if not raw.startswith("---\n") or "\n---\n" not in raw[4:]:
        raise ParamError(f"{path}: missing YAML front matter (--- ... ---)")
    head, body = raw[4:].split("\n---\n", 1)
    try:
        parsed = StrategyFile.model_validate(yaml.safe_load(head))
    except ValidationError as e:
        raise ParamError(f"{path}: {e}") from None
    return Strategy(
        rules=parsed.rules, jev=parsed.jev, body=body, sha256=hashlib.sha256(raw.encode()).hexdigest()
    )


# The search space is fixed BEFORE any backtest (spec §4) and every config tried counts as a trial.
GRID = {
    "rsi_long_max": (35.0, 40.0, 45.0),
    "stop_atr_k": (1.0, 1.5, 2.0),
    "take_profit_r": (1.0, 1.5, 2.0),
    "time_stop_bars": (16, 32),
}
GRID_FIXED = {"pullback_min_atr": 1.0, "pullback_max_atr": 2.5, "direction": "both"}


def grid() -> list[RuleParams]:
    keys = list(GRID)
    return [
        RuleParams(**GRID_FIXED, **dict(zip(keys, vals, strict=True)))
        for vals in itertools.product(*GRID.values())
    ]


def params_hash(p: RuleParams) -> str:
    return hashlib.sha256(json.dumps(p.model_dump(), sort_keys=True).encode()).hexdigest()[:16]
