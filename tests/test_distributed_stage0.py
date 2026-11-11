"""LP variants and MILP: exclusive modes, cost accounting and a negative-price case."""
import numpy as np
import pytest

from fairbattery.stage0_baselines import run_lp, run_milp
from fairbattery.distributed.centralised import solve_day, sufficient_throughput_cost
from fairbattery.env import BatteryCommunity, EnvConfig


def negative_price_env():
    cfg = EnvConfig(annual_kwh=[1000], capacity_kwh=[2], c_rate=0.5,
                    efficiency=0.95, transformer_kw=10, load_noise_sigma=0)
    env = BatteryCommunity(cfg, np.full((1,96), -0.1), np.full((1,96),0.2))
    env.reset([0], np.random.default_rng(7))
    return env


def test_negative_price_cycles_are_real_not_roundoff():
    env = negative_price_env()
    original = solve_day(env,0)
    small = solve_day(env,0,throughput_cost=1e-4)
    assert np.minimum(original.charge,original.discharge).max() > 0.1
    assert np.minimum(small.charge,small.discharge).max() > 0.1
    exact = solve_day(env,0,exclusive=True)
    assert np.minimum(exact.charge,exact.discharge).max() < 1e-6
    assert exact.original_objective > original.original_objective + 1e-3
    assert exact.mip_gap <= 1e-9 + 1e-12


def test_sufficient_fee_eliminates_cycles_but_changes_objective():
    env=negative_price_env()
    fee=sufficient_throughput_cost(env.price,env.eta)
    result=solve_day(env,0,throughput_cost=fee)
    assert np.minimum(result.charge,result.discharge).max() < 1e-6
    assert result.objective == pytest.approx(result.original_objective+result.throughput_fee,abs=1e-8)
    assert result.throughput_fee == pytest.approx(fee*(result.charge+result.discharge).sum())


def test_exclusive_schedule_has_valid_energy_and_original_accounting(env):
    # Includes May's negative-price season, with fixed loads shared by both solves.
    env.reset(np.array([5,120,300]),np.random.default_rng(2025))
    relaxed=run_lp(env)
    exact=run_milp(env)
    c,d,e=(exact[k] for k in ('charge_kwh','discharge_kwh','energy_kwh'))
    assert exact['simultaneous_charge_discharge_steps']==0
    assert e.min() >= -1e-6
    assert (e-env.cap[None,:,None]).max() <= 1e-6
    assert (c-env.pmax[None,:,None]*0.25).max() <= 1e-6
    assert (d-np.minimum(env.loads,env.pmax[None,:,None]*0.25)).max() <= 1e-6
    assert exact['grid_kwh'].min() >= -1e-6
    np.testing.assert_allclose(e,env.cfg.soc_init*env.cap[None,:,None]+np.cumsum(env.eta*c-d/env.eta,axis=2),atol=1e-8)
    score=exact['saving'].sum()-env.cfg.overload_penalty*exact['overload_kwh'].sum()
    assert exact['objective_eur'].sum() == pytest.approx(-score,abs=1e-7)
    assert relaxed['objective_eur'].sum() <= exact['objective_eur'].sum()+1e-6


def test_positive_flat_prices_need_no_cycling():
    env=negative_price_env()
    env.prices_all[:]=0.2
    env.reset([0],np.random.default_rng(7))
    lp=solve_day(env,0); exact=solve_day(env,0,exclusive=True)
    assert lp.original_objective==pytest.approx(exact.original_objective,abs=1e-8)
    assert np.minimum(lp.charge,lp.discharge).max() < 1e-6


def test_transformer_is_soft_not_hard():
    env=negative_price_env()
    env.cfg.transformer_kw=0.1
    env.loads[:]=2.0  # demand cannot be offset by this battery
    result=run_milp(env)
    assert result['overload_kwh'].sum()>0
    assert np.isfinite(result['objective_eur']).all()


@pytest.mark.parametrize('fee', [-1, float('nan'), float('inf')])
def test_invalid_fee_rejected(fee):
    with pytest.raises(ValueError):
        solve_day(negative_price_env(),0,throughput_cost=fee)


def test_corrected_entry_point_returns_cycle_free_lp():
    from fairbattery.stage0_baselines import run_corrected_lp
    env = negative_price_env()
    result = run_corrected_lp(env)
    assert result['simultaneous_charge_discharge_steps'] == 0
    assert result['throughput_cost_eur_per_kwh'] == pytest.approx(
        sufficient_throughput_cost(env.price, env.eta))
    original_cost = -(result['saving'].sum()-env.cfg.overload_penalty*result['overload_kwh'].sum())
    assert result['objective_eur'].sum() == pytest.approx(
        original_cost+result['throughput_fee_eur'].sum(),abs=1e-7)
