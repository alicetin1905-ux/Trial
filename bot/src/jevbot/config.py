"""Configuration: hard risk limits, runtime settings, and secrets.

Risk limits are loaded once into a frozen model. No model output and no runtime code path can
change them. On top of risk.yaml, RISK_CEILINGS are absolute caps written in code: editing the
YAML past a ceiling fails startup.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigError(RuntimeError):
    """Invalid or missing configuration. Always fatal at startup."""


# Absolute caps, independent of risk.yaml. Raising one needs a code change and review.
RISK_CEILINGS: dict[str, float] = {
    "max_risk_per_trade_pct": 2.0,
    "max_notional_x_equity": 3.0,
    "max_leverage": 3,
    "max_open_positions": 1,
    "daily_loss_limit_pct": 5.0,
    "max_drawdown_pct": 20.0,
    "manual_approval_notional_usd": 25_000,
    "max_spread_bps": 20.0,
    "price_band_pct": 3.0,
    "max_order_attempts_per_min": 10,
    "kelly_fraction": 0.5,
    "fixed_risk_pct_until_calibrated": 1.0,
    "calibration_max_ece": 0.1,
}


class RiskLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_risk_per_trade_pct: float = Field(gt=0)
    max_notional_x_equity: float = Field(gt=0)
    max_leverage: int = Field(gt=0)
    max_open_positions: int = Field(gt=0)
    daily_loss_limit_pct: float = Field(gt=0)
    max_drawdown_pct: float = Field(gt=0)
    manual_approval_notional_usd: float = Field(gt=0)
    approval_timeout_s: int = Field(gt=0)
    stale_data_multiple: float = Field(gt=1)
    max_spread_bps: float = Field(gt=0)
    funding_blackout_min: int = Field(ge=0)
    price_band_pct: float = Field(gt=0)
    max_order_attempts_per_min: int = Field(gt=0)
    kill_error_count: int = Field(gt=0)
    kill_error_window_min: int = Field(gt=0)
    kelly_fraction: float = Field(gt=0)
    fixed_risk_pct_until_calibrated: float = Field(gt=0)
    calibration_min_decisions: int = Field(gt=0)
    calibration_max_ece: float = Field(gt=0)

    @model_validator(mode="after")
    def _within_ceilings(self) -> RiskLimits:
        over = [k for k, cap in RISK_CEILINGS.items() if getattr(self, k) > cap]
        if over:
            msgs = (f"{k}={getattr(self, k)} exceeds ceiling {RISK_CEILINGS[k]}" for k in over)
            raise ValueError(", ".join(msgs))
        if self.fixed_risk_pct_until_calibrated > self.max_risk_per_trade_pct:
            raise ValueError("fixed_risk_pct_until_calibrated exceeds max_risk_per_trade_pct")
        return self


def _validation_message(e: ValidationError) -> str:
    parts = []
    for err in e.errors():
        loc = ".".join(str(x) for x in err["loc"])
        parts.append(f"{loc}: {err['msg']}" if loc else err["msg"])
    return "; ".join(parts)


def _read_yaml(path: Path) -> tuple[dict, bytes]:
    raw = Path(path).read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a mapping at top level")
    return data, raw


def load_risk_limits(path: Path) -> tuple[RiskLimits, str]:
    """Return (limits, sha256 of the file bytes). The hash is journaled with every decision."""
    data, raw = _read_yaml(path)
    try:
        limits = RiskLimits.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path}: {_validation_message(e)}") from None
    return limits, hashlib.sha256(raw).hexdigest()


# --- runtime settings ------------------------------------------------------------------------


class _Strict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class MarketSettings(_Strict):
    symbol: str
    category: Literal["linear"]
    entry_timeframe_min: int = Field(gt=0)
    trend_timeframes_min: tuple[int, ...]


class JevSettings(_Strict):
    model: str
    pinned_model: str
    live_timeout_s: float = Field(gt=0, le=5)
    backtest_max_concurrency: int = Field(gt=0, le=64)


class OpusSettings(_Strict):
    model: str
    effort: Literal["low", "medium", "high", "xhigh", "max"]
    monthly_cap_usd: float = Field(gt=0)
    per_run_cap_usd: float = Field(gt=0)
    max_candidates_per_run: int = Field(gt=0, le=10)

    @model_validator(mode="after")
    def _caps(self) -> OpusSettings:
        if self.per_run_cap_usd > self.monthly_cap_usd:
            raise ValueError("per_run_cap_usd exceeds monthly_cap_usd")
        return self


class PathSettings(_Strict):
    data_dir: Path
    db_path: Path
    active_dir: Path
    staging_dir: Path
    reports_dir: Path
    halt_file: Path


class Settings(_Strict):
    # Only paper is accepted. Live is unlocked outside this loader (spec §17, architecture §8).
    mode: Literal["paper"]
    market: MarketSettings
    jev: JevSettings
    opus: OpusSettings
    paths: PathSettings


def load_settings(path: Path) -> Settings:
    data, _ = _read_yaml(path)
    try:
        return Settings.model_validate(data)
    except ValidationError as e:
        raise ConfigError(f"{path}: {_validation_message(e)}") from None


# --- secrets ---------------------------------------------------------------------------------


class Secrets(BaseSettings):
    """Read from the environment / .env only. Values are SecretStr, so they never print."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    TYPESAFE_API_KEY: SecretStr | None = None
    ANTHROPIC_API_KEY: SecretStr | None = None
    BYBIT_DEMO_KEY: SecretStr | None = None
    BYBIT_DEMO_SECRET: SecretStr | None = None
    TELEGRAM_BOT_TOKEN: SecretStr | None = None
    TELEGRAM_CHAT_ID: SecretStr | None = None
    DASHBOARD_USER: SecretStr | None = None
    DASHBOARD_PASS: SecretStr | None = None

    def require(self, *names: str) -> None:
        missing = [n for n in names if not self._value(n)]
        if missing:
            raise ConfigError(f"missing secrets: {', '.join(missing)} (set them in .env)")

    def _value(self, name: str) -> str:
        v = getattr(self, name)
        return v.get_secret_value() if v is not None else ""

    def values(self) -> list[str]:
        return [v for n in type(self).model_fields if len(v := self._value(n)) >= 4]


class RedactSecrets(logging.Filter):
    """Replaces any secret value in a log record, including its traceback, with ***.

    Attach it to every *handler*: a logger-level filter does not see records from child loggers.
    """

    _formatter = logging.Formatter()

    def __init__(self, secrets: Secrets) -> None:
        super().__init__()
        # Longest first, so a secret that contains another is fully masked.
        self._values = sorted(secrets.values(), key=len, reverse=True)

    def _redact(self, text: str) -> str:
        for v in self._values:
            text = text.replace(v, "***")
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        if not self._values:
            return True
        record.msg, record.args = self._redact(record.getMessage()), None
        if record.exc_info and not record.exc_text:
            # Formatter.format() reuses exc_text when set, so the raw traceback is never rendered.
            record.exc_text = self._formatter.formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redact(record.stack_info)
        return True
