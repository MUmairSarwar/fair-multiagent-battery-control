import numpy as np

from fairbattery.baselines import PriceRulePolicy, run_lp
from fairbattery.metrics import baseline_cost, rollout, summarise


def test_lp_beats_rule_and_respects_limits(env, rng):
    days = np.array([5, 120, 300])
    env.reset(days, rng)
    loads = env.loads.copy()
    bc = baseline_cost(env)
    rule = summarise(rollout(env, PriceRulePolicy(), days, rng, loads=loads), env, bc)
    env.reset(days, rng, loads=loads)
    lp_res = run_lp(env)
    lp = summarise(lp_res, env, bc)
    assert lp["net_benefit_eur"] >= rule["net_benefit_eur"] - 1e-6
    assert lp["net_benefit_eur"] >= -1e-6  # doing nothing is feasible
    assert lp["peak_kw"] <= env.cfg.transformer_kw + 1e-6
    assert (lp_res["grid_kwh"] >= -1e-9).all()


def test_rule_overloads_the_transformer(env, rng):
    days = np.arange(7)
    res = rollout(env, PriceRulePolicy(), days, rng)
    assert res["overload_kwh"].sum() > 0
