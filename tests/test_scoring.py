"""Phase 4: scoring engine, incl. the blueprint section 11 acceptance tests."""
import numpy as np
import pandas as pd
import pytest

import scoring.score as sc
from scoring.data import PILOT_IDS, spec
from scoring.score import Profile, effective_weights, percentile, score


def test_weights_are_versioned_design_defaults():
    s = spec()
    assert s["version"] == "v4"
    assert [spec(v)["version"] for v in ("v1", "v2", "v3")] == ["v1", "v2", "v3"]  # previous versions kept
    for ind, prof in s["industries"].items():
        assert sum(prof["weights"].values()) == 100, ind
        assert set(prof["weights"]) <= set(s["dimensions"]), ind
        assert set(prof["weights"]) == set(prof["indicators"]), ind
        assert set(prof["mandatory_dimensions"]) <= set(prof["weights"]), ind


def test_percentile_midrank_and_direction():
    ref = np.array([1.0, 2.0, 3.0, 4.0])
    assert percentile(4.0, ref) == 87.5 and percentile(0.0, ref) == 0.0 and percentile(9, ref) == 100
    assert np.isnan(percentile(float("nan"), ref))
    t = sc.indicator_table(Profile("manufacturing"))
    cost = t[(t.indicator == "rural_wage_nonagri") & t.score.notna()]
    hi, lo = cost.loc[cost.value.idxmax()], cost.loc[cost.value.idxmin()]
    assert hi.score < lo.score  # higher wage cost -> lower score


def test_score_bounds_and_ranks():
    r = score(Profile("manufacturing"))
    assert r.ranking.score.between(0, 100).all()
    assert list(r.ranking["rank"]) == list(range(1, len(r.ranking) + 1))


# --- section 11: "Variable selected with no observed data" -> exclude or mark unavailable, never score as observed
def test_missing_value_is_not_filled_and_is_disclosed():
    r = score(Profile("manufacturing"))
    tg = r.indicators[(r.indicators.state_id == "TG") & (r.indicators.indicator == "rural_wage_nonagri")]
    assert tg.score.isna().all() and not tg.observed.any()
    assert tg.w_ind_eff.isna().all() or (tg.w_ind_eff.fillna(0) == 0).all()
    assert "Rural non-farm wage" in r.ranking.set_index("state_id").disclosure["TG"]
    # renormalisation: TG's observed cost-indicator weights still sum to 1
    obs = r.indicators[(r.indicators.state_id == "TG") & (r.indicators.dimension == "cost") & r.indicators.observed]
    assert obs.w_ind_eff.sum() == pytest.approx(1.0)


def test_missing_value_never_gets_a_favourable_fill(monkeypatch):
    """Removing a state's best indicator must not raise its dimension score above the observed-only average."""
    base = score(Profile("retail"), with_stability=False)
    orig = sc.indicator_table

    def drop(p, version="v1"):
        t = orig(p, version)
        m = (t.state_id == "KA") & (t.indicator == "mpce_urban")
        t.loc[m, ["value", "score"]] = np.nan
        return t
    monkeypatch.setattr(sc, "indicator_table", drop)
    r = score(Profile("retail"), with_stability=False)
    ka = r.indicators[(r.indicators.state_id == "KA") & (r.indicators.dimension == "demand") & r.indicators.observed]
    expect = (ka.score * ka.w_ind).sum() / ka.w_ind.sum()
    assert r.ranking.set_index("state_id").dim_demand["KA"] == pytest.approx(expect)
    assert "Urban household spending" in r.ranking.set_index("state_id").disclosure["KA"]
    assert base.ranking.set_index("state_id").disclosure["KA"] == ""


# --- section 11: "State missing mandatory ..." and "Insufficient eligible states"
def test_state_without_mandatory_dimension_is_excluded(monkeypatch):
    orig = sc.indicator_table

    def wipe(p, version="v1"):
        t = orig(p, version)
        t.loc[(t.state_id == "RJ") & (t.dimension == "infrastructure"), ["value", "score"]] = np.nan
        return t
    monkeypatch.setattr(sc, "indicator_table", wipe)
    r = score(Profile("manufacturing"), with_stability=False)
    assert "RJ" in set(r.excluded.state_id) and "RJ" not in set(r.ranking.state_id)


def test_insufficient_states_gives_no_ranking():
    r = score(Profile("manufacturing", states=["MH"]))
    assert r.ranking.empty and "Fewer than two" in r.message


# --- section 11: "Different user priorities" -> ranking changes only through disclosed, versioned weights
def test_priorities_change_weights_transparently():
    w0, _, n0 = effective_weights(Profile("retail"))
    w1, _, n1 = effective_weights(Profile("retail", priorities=["cost"]))
    assert w1["cost"] > w0["cost"] and abs(sum(w1.values()) - 1) < 1e-9
    assert any("cost" in n.lower() for n in n1) and not n0
    a = score(Profile("retail", priorities=["cost"]), with_stability=False).ranking
    b = score(Profile("retail", priorities=["cost"]), with_stability=False).ranking
    pd.testing.assert_frame_equal(a, b)  # deterministic


def test_policy_priority_reweights_the_policy_dimension():
    """v3: 'policy' maps to a dimension scored from official assessments (BRAP, LEADS, EPI, Startup Ranking)."""
    base = score(Profile("logistics"), with_stability=False)
    pol = score(Profile("logistics", priorities=["policy"]), with_stability=False)
    assert pol.dim_weights["policy"] > base.dim_weights["policy"]
    assert any("Priority 'policy'" in n for n in pol.notes)
    assert not np.allclose(base.ranking.set_index("state_id").score.sort_index(),
                           pol.ranking.set_index("state_id").score.sort_index())

def test_too_many_priorities_rejected():
    with pytest.raises(ValueError):
        effective_weights(Profile("retail", priorities=["demand", "cost", "labour", "growth"]))


def test_operational_need_adds_its_indicators_and_is_disclosed():
    _, i0, _ = effective_weights(Profile("manufacturing"))
    w1, i1, notes = effective_weights(Profile("manufacturing", needs=["digital_first"]))
    assert "internet_subs_per_100" not in i0["infrastructure"]
    assert i1["infrastructure"]["internet_subs_per_100"] == 2  # added at weight 1, then x2 for the need
    assert any("adds internet_subs_per_100" in n for n in notes)
    t = score(Profile("manufacturing", needs=["digital_first"]), with_stability=False).indicators
    assert "internet_subs_per_100" in set(t.indicator)


def test_frozen_reference_scores_do_not_depend_on_comparison_set():
    full = score(Profile("it_services"), with_stability=False).ranking.set_index("state_id").score
    sub = score(Profile("it_services", states=["KA", "TG", "MH"]), with_stability=False).ranking.set_index("state_id").score
    assert np.allclose(full[sub.index], sub)


def test_exclude_current_state():
    r = score(Profile("retail", current_state="UP", exclude_current=True), with_stability=False)
    assert "UP" not in set(r.ranking.state_id)


def test_no_capital_gives_v3_scores():
    """v4 only adds the capital tilt: without an investment entered, every industry scores exactly as in v3."""
    for ind in spec()["industries"]:
        a = score(Profile(ind), version="v3", with_stability=False).ranking.set_index("state_id").score
        b = score(Profile(ind), with_stability=False).ranking.set_index("state_id").score
        assert np.allclose(a[b.index], b), ind


@pytest.mark.parametrize("cap,band", [(0.5, "small"), (9.99, "small"), (10, "mid"), (49, "mid"), (50, "upper_mid"),
                                      (100, "large"), (5000, "large")])
def test_capital_band_edges(cap, band):
    assert sc.capital_band(cap)["key"] == band
    assert sc.capital_band(None) is None and sc.capital_band(0) is None


def test_capital_tilts_weights_as_specified():
    s = spec()
    base, _, _ = effective_weights(Profile("manufacturing"))
    for b in s["capital"]["bands"]:
        cap = 1 if b["upper"] == 10 else (b["upper"] or 1000) - 1
        w, _, notes = effective_weights(Profile("manufacturing", capital_rs_crore=cap))
        raw = {d: v * b["tilt"].get(d, 1) for d, v in base.items()}
        tot = sum(raw.values())
        assert all(w[d] == pytest.approx(raw[d] / tot) for d in w), b["key"]
        assert any("design default" in n or "unchanged" in n for n in notes), notes


def test_capital_tilt_skips_missing_dimension_and_what_if():
    w, _, notes = effective_weights(Profile("healthcare", capital_rs_crore=500))   # healthcare has no ecosystem
    assert "ecosystem" not in w and any("skipped" in n for n in notes)
    dw = {"demand": 10, "infrastructure": 30, "outlook": 20, "labour": 10, "ecosystem": 10, "cost": 10, "policy": 10}
    w, _, notes = effective_weights(Profile("manufacturing", capital_rs_crore=500, dimension_weights=dw))
    assert w["infrastructure"] == pytest.approx(0.3) and any("what-if" in n for n in notes)


def test_capital_fit_traces_to_asi_values():
    r = score(Profile("manufacturing", capital_rs_crore=8), with_stability=False)
    fit = sc.capital_fit(r)
    assert fit["band"]["key"] == "small" and fit["shown"]
    lat = sc.latest(["asi_fixed_capital", "asi_factories", "asi_gva"]).set_index(["state_id", "feature_code"]).value
    for b in fit["benchmarks"].itertuples():
        fc, nf, gva = lat[(b.state_id, "asi_fixed_capital")], lat[(b.state_id, "asi_factories")], lat[(b.state_id, "asi_gva")]
        assert b.plant_cr == pytest.approx(fc / 100 / nf)          # Rs lakh -> Rs crore, converted once in ETL
        assert b.gva_per_100 == pytest.approx(gva / fc * 100)
        assert b.implied_gva_cr == pytest.approx(8 * gva / fc)
    assert set(fit["benchmarks"].state_id) == set(r.ranking.state_id)
    # benchmarks never enter a score
    assert not r.indicators.indicator.str.startswith("asi_fixed_capital").any()
    it = sc.capital_fit(score(Profile("it_services", capital_rs_crore=8), with_stability=False))
    assert not it["shown"] and it["benchmarks"].empty
    assert sc.capital_fit(score(Profile("retail"), with_stability=False)) is None


# --- section 11: "Sector lacks defensible target" -> ranking-only and no fabricated forecast
def test_no_forecast_factor_for_ranking_only_sectors():
    for ind in ["retail", "it_services", "logistics", "healthcare", "hospitality"]:
        r = score(Profile(ind), with_stability=False)
        assert not r.indicators.indicator.str.startswith("forecast").any()


def test_forecast_factor_only_when_validated():
    from scoring.data import registry
    r = score(Profile("manufacturing"), with_stability=False)
    has = r.indicators.indicator.eq("forecast_manufacturing").any()
    assert has == (registry()["manufacturing"]["status"] == "validated")
    if has:  # Gujarat has no forecast -> missing, not filled
        gj = r.indicators[(r.indicators.state_id == "GJ") & (r.indicators.indicator == "forecast_manufacturing")]
        assert gj.score.isna().all()


def test_stability_and_counter_explanation():
    r = score(Profile("manufacturing", priorities=["infrastructure", "labour"],
                      needs=["power_intensive", "logistics_heavy"]))
    st = r.stability
    assert set(st.state_id) == set(r.ranking.state_id)
    assert st.share_top3.between(0, 1).all() and (st.best_rank <= st["rank"]).all()
    assert r.counter and r.ranking.state.iloc[1] in r.counter


def test_all_industries_score_all_pilots():
    for ind in spec()["industries"]:
        r = score(Profile(ind), with_stability=False)
        assert len(r.ranking) + len(r.excluded) == len(PILOT_IDS)


def test_whatif_override_sets_final_dimension_weights():
    base = Profile("manufacturing", priorities=["infrastructure"], needs=["power_intensive"])
    w, _, _ = effective_weights(base)
    w2, inds2, notes = effective_weights(Profile(**{**base.__dict__, "dimension_weights": {k: v * 100 for k, v in w.items()}}))
    assert all(abs(w[k] - w2[k]) < 1e-9 for k in w)          # same starting point, no double multiplication
    assert inds2["infrastructure"]["power_shortage_pct"] == 4  # indicator boost from the need still applies
    a = score(base, with_stability=False).ranking.set_index("state_id").score
    b = score(Profile(**{**base.__dict__, "dimension_weights": {k: v * 100 for k, v in w.items()}}),
              with_stability=False).ranking.set_index("state_id").score
    assert np.allclose(a.sort_index(), b.sort_index())
