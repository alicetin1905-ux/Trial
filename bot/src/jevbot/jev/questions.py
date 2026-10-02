"""questions.yaml -> TypeSafe SDK question objects, and the state Jev is asked about."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml
from typesafe_sdk import Choice, Noul, Score

QUESTION_NAMES = ("regime", "direction", "buy_pressure_real", "setup_quality", "risk_state")

# combine.py reads these labels, so a questions file may reword descriptions but not drop them.
REQUIRED = {
    "regime": ("choice", {"trending_up", "trending_down"}),
    "direction": ("choice", {"up", "down"}),
    "buy_pressure_real": ("noul", set()),
    "setup_quality": ("score", set()),
    "risk_state": ("choice", {"normal", "extreme"}),
}
SETUP_QUALITY_LEVELS = 5  # 0..4

SIDE_LABEL = {1: "long", -1: "short", 0: "none"}


class QuestionsError(ValueError):
    pass


@dataclass(frozen=True)
class Questions:
    questions: dict[str, Choice | Score | Noul]
    legend: dict

    def wire(self) -> dict:
        return {k: v.model_dump() for k, v in self.questions.items()}


def _build(name: str, spec: dict) -> Choice | Score | Noul:
    kind = spec.get("type")
    want, labels = REQUIRED[name]
    if kind != want:
        raise QuestionsError(f"{name}: expected type {want}, got {kind}")
    if kind == "choice":
        crit = {str(k): v for k, v in spec["criteria"].items()}
        missing = labels - set(crit)
        if missing:
            raise QuestionsError(f"{name}: missing required labels {sorted(missing)}")
        return Choice(instructions=spec.get("instructions"), criteria=crit)
    if kind == "score":
        if len(spec["criteria"]) != SETUP_QUALITY_LEVELS:
            raise QuestionsError(f"{name}: needs exactly {SETUP_QUALITY_LEVELS} levels")
        return Score(instructions=spec.get("instructions"), criteria=list(spec["criteria"]))
    # YAML reads `true:` / `false:` as booleans; the API wants the strings.
    crit = {str(k).lower(): v for k, v in (spec.get("criteria") or {}).items()}
    return Noul(instructions=spec.get("instructions"), criteria=crit or None)


def load_questions(path: Path) -> Questions:
    data = yaml.safe_load(Path(path).read_text())
    qs = data.get("questions") or {}
    if set(qs) != set(QUESTION_NAMES):
        raise QuestionsError(f"questions must be exactly {QUESTION_NAMES}, got {sorted(qs)}")
    return Questions(
        questions={n: _build(n, qs[n]) for n in QUESTION_NAMES},
        legend={str(k): v for k, v in (data.get("legend") or {}).items()},
    )


def build_state(snapshot: dict[str, float], side: int, legend: dict) -> dict:
    """The JSON state for one call. Key order is fixed, so identical inputs give identical bytes."""
    return {
        "legend": dict(sorted(legend.items())),
        "candidate_side": SIDE_LABEL[side],
        "features": dict(sorted(snapshot.items())),
    }
