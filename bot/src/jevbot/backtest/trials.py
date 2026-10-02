"""Append-only log of every configuration ever evaluated. It feeds the deflated Sharpe (spec §9)."""

from __future__ import annotations

import json
import time
from pathlib import Path


class TrialLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a") as f:  # append only: earlier lines are never rewritten
            f.write(json.dumps(record | {"ts": time.time()}, sort_keys=True) + "\n")
            f.flush()

    def read(self) -> list[dict]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]

    def trial_sharpes(self) -> list[float]:
        return [r["sharpe_daily"] for r in self.read() if "sharpe_daily" in r]
