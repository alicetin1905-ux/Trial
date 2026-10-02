import hashlib
import logging
from pathlib import Path

import pytest
import yaml

from jevbot.config import (
    RISK_CEILINGS,
    ConfigError,
    RedactSecrets,
    RiskLimits,
    Secrets,
    load_risk_limits,
    load_settings,
)

ROOT = Path(__file__).resolve().parents[1]
RISK_YAML = ROOT / "config" / "risk.yaml"
SETTINGS_YAML = ROOT / "config" / "settings.yaml"


def _risk_dict() -> dict:
    return yaml.safe_load(RISK_YAML.read_text())


def _write(tmp_path: Path, name: str, data: dict) -> Path:
    p = tmp_path / name
    p.write_text(yaml.safe_dump(data))
    return p


# --- risk.yaml -------------------------------------------------------------------------------


def test_shipped_risk_yaml_matches_approved_spec():
    limits, _ = load_risk_limits(RISK_YAML)
    assert limits.max_risk_per_trade_pct == 1.0
    assert limits.daily_loss_limit_pct == 3.0
    assert limits.max_drawdown_pct == 10.0
    assert limits.max_leverage == 2
    assert limits.max_open_positions == 1
    assert limits.manual_approval_notional_usd == 2500
    assert limits.approval_timeout_s == 300
    assert limits.kelly_fraction == 0.25


def test_risk_limits_are_frozen():
    limits, _ = load_risk_limits(RISK_YAML)
    with pytest.raises(Exception):  # noqa: B017 - pydantic raises ValidationError on frozen assignment
        limits.max_leverage = 50


def test_risk_sha256_is_hash_of_file_bytes():
    _, sha = load_risk_limits(RISK_YAML)
    assert sha == hashlib.sha256(RISK_YAML.read_bytes()).hexdigest()


def test_unknown_risk_key_fails_startup(tmp_path):
    d = _risk_dict() | {"allow_model_override": True}
    with pytest.raises(ConfigError, match="allow_model_override"):
        load_risk_limits(_write(tmp_path, "risk.yaml", d))


def test_missing_risk_key_fails_startup(tmp_path):
    d = _risk_dict()
    del d["max_drawdown_pct"]
    with pytest.raises(ConfigError, match="max_drawdown_pct"):
        load_risk_limits(_write(tmp_path, "risk.yaml", d))


@pytest.mark.parametrize("key", sorted(RISK_CEILINGS))
def test_value_above_hard_ceiling_fails_even_if_file_is_edited(tmp_path, key):
    d = _risk_dict()
    d[key] = RISK_CEILINGS[key] * 2 + 1
    with pytest.raises(ConfigError, match=key):
        load_risk_limits(_write(tmp_path, "risk.yaml", d))


@pytest.mark.parametrize("key", ["max_risk_per_trade_pct", "max_leverage", "daily_loss_limit_pct"])
def test_non_positive_limits_rejected(tmp_path, key):
    d = _risk_dict()
    d[key] = 0
    with pytest.raises(ConfigError, match=key):
        load_risk_limits(_write(tmp_path, "risk.yaml", d))


def test_risk_limits_type_is_exported():
    limits, _ = load_risk_limits(RISK_YAML)
    assert isinstance(limits, RiskLimits)


# --- settings.yaml ---------------------------------------------------------------------------


def test_shipped_settings_load_in_paper_mode():
    s = load_settings(SETTINGS_YAML)
    assert s.mode == "paper"
    assert s.market.symbol == "BTCUSDT"
    assert s.market.entry_timeframe_min == 15
    assert s.opus.model == "claude-opus-5-5"
    assert s.opus.monthly_cap_usd == 5.0
    assert s.jev.live_timeout_s == 1.5


def test_live_mode_is_rejected(tmp_path):
    d = yaml.safe_load(SETTINGS_YAML.read_text()) | {"mode": "live"}
    with pytest.raises(ConfigError, match="mode"):
        load_settings(_write(tmp_path, "settings.yaml", d))


def test_opus_per_run_cap_cannot_exceed_monthly_cap(tmp_path):
    d = yaml.safe_load(SETTINGS_YAML.read_text())
    d["opus"]["per_run_cap_usd"] = 6.0
    with pytest.raises(ConfigError, match="per_run_cap_usd"):
        load_settings(_write(tmp_path, "settings.yaml", d))


def test_unknown_settings_key_rejected(tmp_path):
    d = yaml.safe_load(SETTINGS_YAML.read_text()) | {"max_leverage": 20}
    with pytest.raises(ConfigError, match="max_leverage"):
        load_settings(_write(tmp_path, "settings.yaml", d))


# --- secrets ---------------------------------------------------------------------------------

SECRET = "sk-test-SUPERSECRET-123456"


def test_secret_values_never_appear_in_repr(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    s = Secrets(_env_file=None)
    assert SECRET not in repr(s)
    assert SECRET not in str(s)
    assert s.TYPESAFE_API_KEY.get_secret_value() == SECRET


def test_require_names_missing_secrets_without_leaking_present_ones(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    monkeypatch.delenv("BYBIT_DEMO_KEY", raising=False)
    s = Secrets(_env_file=None)
    with pytest.raises(ConfigError) as e:
        s.require("TYPESAFE_API_KEY", "BYBIT_DEMO_KEY")
    assert "BYBIT_DEMO_KEY" in str(e.value)
    assert SECRET not in str(e.value)


def test_redaction_filter_hides_secrets_in_message_and_args(monkeypatch, caplog):
    monkeypatch.setenv("TYPESAFE_API_KEY", SECRET)
    s = Secrets(_env_file=None)
    log = logging.getLogger("jevbot.test.redact")
    log.addFilter(RedactSecrets(s))
    with caplog.at_level(logging.INFO, logger="jevbot.test.redact"):
        log.info("header was Bearer %s", SECRET)
        log.info(f"inline {SECRET} leak")
    text = caplog.text
    assert SECRET not in text
    assert text.count("***") == 2


def test_redaction_ignores_empty_secrets(monkeypatch, caplog):
    for name in Secrets.model_fields:
        monkeypatch.delenv(name, raising=False)
    s = Secrets(_env_file=None)
    log = logging.getLogger("jevbot.test.redact_empty")
    log.addFilter(RedactSecrets(s))
    with caplog.at_level(logging.INFO, logger="jevbot.test.redact_empty"):
        log.info("nothing to hide")
    assert "nothing to hide" in caplog.text


def test_redaction_covers_exception_tracebacks(monkeypatch, caplog):
    monkeypatch.setenv("BYBIT_DEMO_SECRET", SECRET)
    s = Secrets(_env_file=None)
    log = logging.getLogger("jevbot.test.redact_exc")
    log.addFilter(RedactSecrets(s))
    with caplog.at_level(logging.ERROR, logger="jevbot.test.redact_exc"):
        try:
            raise RuntimeError(f"401 from https://api-demo.bybit.com/?api_secret={SECRET}")
        except RuntimeError:
            log.exception("order failed")
    assert SECRET not in caplog.text
    assert "api_secret=***" in caplog.text
