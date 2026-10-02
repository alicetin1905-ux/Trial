import pandas as pd
import pytest
from conftest import make_bars15

from jevbot.state.snapshot import WARMUP_BARS, compute_features
from jevbot.strategy.params import ParamError, RuleParams, grid, load_strategy_md, params_hash
from jevbot.strategy.rules import candidates, exit_levels, trend_dir, trend_flipped

STRATEGY_MD = """---
rules:
  pullback_min_atr: 1.0
  pullback_max_atr: 2.5
  rsi_long_max: 40
  stop_atr_k: 1.5
  take_profit_r: 1.5
  time_stop_bars: 32
  direction: both
---
# Trend + pullback

Human-readable description of the strategy.
"""


def _rules(**kw) -> RuleParams:
    base = dict(
        pullback_min_atr=1.0,
        pullback_max_atr=2.5,
        rsi_long_max=40,
        stop_atr_k=1.5,
        take_profit_r=1.5,
        time_stop_bars=32,
        direction="both",
    )
    return RuleParams(**(base | kw))


def _feat(**kw) -> pd.DataFrame:
    row = dict(
        t=0,
        ema50_slope_1h_atr=0.3,
        dist_ema50_4h_atr=1.0,
        pullback_6h_atr1h=1.5,
        bounce_6h_atr1h=0.2,
        rsi14_15m=30.0,
    )
    return pd.DataFrame([row | kw])


# --- params ----------------------------------------------------------------------------------


def test_load_strategy_md_front_matter(tmp_path):
    p = tmp_path / "strategy.md"
    p.write_text(STRATEGY_MD)
    s = load_strategy_md(p)
    assert s.rules.stop_atr_k == 1.5 and s.rules.direction == "both"
    assert "Human-readable" in s.body


def test_unknown_rule_key_rejected(tmp_path):
    p = tmp_path / "strategy.md"
    p.write_text(STRATEGY_MD.replace("direction: both", "direction: both\n  max_leverage: 50"))
    with pytest.raises(ParamError, match="max_leverage"):
        load_strategy_md(p)


def test_missing_front_matter_rejected(tmp_path):
    p = tmp_path / "strategy.md"
    p.write_text("# no front matter\n")
    with pytest.raises(ParamError):
        load_strategy_md(p)


def test_pullback_range_must_be_ordered():
    with pytest.raises(ParamError):
        RuleParams.checked(**_rules().model_dump() | {"pullback_min_atr": 3.0})


def test_grid_is_54_unique_configs_with_stable_hashes():
    g = grid()
    assert len(g) == 54
    hashes = {params_hash(p) for p in g}
    assert len(hashes) == 54
    assert params_hash(g[0]) == params_hash(RuleParams(**g[0].model_dump()))


# --- trend & candidates ----------------------------------------------------------------------


def test_trend_requires_1h_slope_and_4h_side_to_agree():
    assert trend_dir(_feat()).iloc[0] == 1
    assert trend_dir(_feat(ema50_slope_1h_atr=-0.3, dist_ema50_4h_atr=-1.0)).iloc[0] == -1
    assert trend_dir(_feat(ema50_slope_1h_atr=0.3, dist_ema50_4h_atr=-1.0)).iloc[0] == 0


def test_long_candidate_in_uptrend_pullback_with_low_rsi():
    assert candidates(_feat(), _rules()).iloc[0] == 1


@pytest.mark.parametrize(
    "kw",
    [
        {"pullback_6h_atr1h": 0.5},  # too shallow
        {"pullback_6h_atr1h": 3.0},  # too deep: structure broken
        {"rsi14_15m": 45.0},  # not oversold enough
        {"dist_ema50_4h_atr": -1.0},  # trends disagree
    ],
)
def test_no_long_candidate_when_a_condition_fails(kw):
    assert candidates(_feat(**kw), _rules()).iloc[0] == 0


def test_short_candidate_mirrors_long():
    f = _feat(
        ema50_slope_1h_atr=-0.3,
        dist_ema50_4h_atr=-1.0,
        bounce_6h_atr1h=1.5,
        pullback_6h_atr1h=0.1,
        rsi14_15m=70,
    )
    assert candidates(f, _rules()).iloc[0] == -1
    assert candidates(f, _rules(direction="long")).iloc[0] == 0


def test_pullback_bounds_are_inclusive():
    assert candidates(_feat(pullback_6h_atr1h=1.0), _rules()).iloc[0] == 1
    assert candidates(_feat(pullback_6h_atr1h=2.5), _rules()).iloc[0] == 1


def test_candidates_on_real_feature_frame_are_rare_and_two_sided():
    f = compute_features(make_bars15(WARMUP_BARS + 30 * 96, seed=7))
    c = candidates(f, _rules())
    assert set(c.unique()) <= {-1, 0, 1}
    assert 0 < (c != 0).mean() < 0.2


# --- exits -----------------------------------------------------------------------------------


def test_exit_levels_long_and_short():
    sl, tp = exit_levels(entry=100.0, side=1, atr=2.0, rules=_rules())
    assert (sl, tp) == (pytest.approx(97.0), pytest.approx(104.5))
    sl, tp = exit_levels(entry=100.0, side=-1, atr=2.0, rules=_rules())
    assert (sl, tp) == (pytest.approx(103.0), pytest.approx(95.5))


def test_exit_levels_reject_bad_atr():
    with pytest.raises(ValueError):
        exit_levels(entry=100.0, side=1, atr=0.0, rules=_rules())


def test_trend_flip_invalidates_position():
    assert trend_flipped(side=1, trend=-1)
    assert not trend_flipped(side=1, trend=0)  # neutral is not a flip
    assert not trend_flipped(side=-1, trend=-1)


JEV_BLOCK = """jev:
  min_p_regime: 0.5
  min_p_direction: 0.4
  min_pressure: 0.5
  min_setup_quality: 2.0
  max_p_risk_extreme: 0.2
  w_regime: 1.0
  w_direction: 1.0
  w_pressure: 0.5
  w_quality: 1.0
  w_risk_normal: 0.5
  intercept: -2.0
  p_cutoff: 0.55
"""


def test_strategy_md_jev_section_is_optional_and_validated(tmp_path):
    p = tmp_path / "strategy.md"
    p.write_text(STRATEGY_MD)
    assert load_strategy_md(p).jev is None
    p.write_text(STRATEGY_MD.replace("---\n# Trend", JEV_BLOCK + "---\n# Trend"))
    s = load_strategy_md(p)
    assert s.jev is not None and s.jev.p_cutoff == 0.55
    p.write_text(
        STRATEGY_MD.replace(
            "---\n# Trend", JEV_BLOCK.replace("p_cutoff: 0.55", "p_cutoff: 1.5") + "---\n# Trend"
        )
    )
    with pytest.raises(ParamError, match="p_cutoff"):
        load_strategy_md(p)
