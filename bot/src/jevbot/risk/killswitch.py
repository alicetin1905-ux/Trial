"""Kill switch (spec §8): cancel everything, flatten reduce-only, write HALTED, alert, refuse orders.

Fail-safe: if the exchange calls fail, the HALTED file is still written and the alert says the
flatten FAILED, so a human acts. Only an explicit, attributed human reset clears it.
"""

from __future__ import annotations

import json
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from jevbot.config import RiskLimits


class Broker(Protocol):
    def cancel_all(self) -> None: ...
    def position(self) -> tuple[int, float]: ...  # (side, units); (0, 0.0) when flat
    def close_reduce_only(self, side: int, units: float) -> None: ...


class KillSwitch:
    def __init__(self, broker: Broker, halt_path: Path, alert: Callable[[str], None]) -> None:
        self.broker, self.halt_path, self.alert = broker, Path(halt_path), alert

    def is_halted(self) -> bool:
        return self.halt_path.exists()

    def fire(self, reason: str, now_ms: int) -> None:
        first = not self.is_halted()
        # Write HALTED first: from this instant every order path refuses to trade.
        if first:
            self.halt_path.write_text(json.dumps({"reason": reason, "at_ms": now_ms}))
        problems = []
        try:
            self.broker.cancel_all()
        except Exception as e:  # keep going: flattening matters more than cancelling
            problems.append(f"cancel_all: {e}")
        try:
            side, units = self.broker.position()
            if side != 0 and units > 0:
                self.broker.close_reduce_only(side, units)
        except Exception as e:
            problems.append(f"flatten: {e}")
        if problems:
            self.alert(f"🛑 KILL ({reason}) — FLATTEN FAILED, ACT NOW: {'; '.join(problems)}")
        elif first:
            self.alert(
                f"🛑 KILL ({reason}): orders cancelled, position flat, trading halted until manual reset"
            )

    def reset(self, confirmed_by: str) -> None:
        if not confirmed_by:
            raise PermissionError("a halt can only be cleared by a named human")
        if self.is_halted():
            self.halt_path.unlink()
            self.alert(f"✅ halt cleared by {confirmed_by}")


class ErrorTracker:
    """Fires on_trip('order_errors') at kill_error_count errors within kill_error_window_min."""

    def __init__(self, limits: RiskLimits, on_trip: Callable[[str], None]) -> None:
        self.n = limits.kill_error_count
        self.window = limits.kill_error_window_min * 60_000
        self.on_trip = on_trip
        self.times: deque[int] = deque()

    def record(self, now_ms: int) -> None:
        self.times.append(now_ms)
        while self.times and now_ms - self.times[0] > self.window:
            self.times.popleft()
        if len(self.times) >= self.n:
            self.on_trip("order_errors")
