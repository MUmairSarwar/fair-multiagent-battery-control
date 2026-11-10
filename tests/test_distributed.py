import numpy as np
import pytest

from fairbattery import data
from fairbattery.baselines import solve_day_lp
from fairbattery.distributed import dual_decomposition, from_env, sharing_admm, solve_centralized
from fairbattery.distributed.admm import prox_g


@pytest.fixture
def day(env, rng):
    """One test day with a congested transformer (day index 10 of the test set)."""
    _, test_days = data.split_days(len(data.load_days()[0]))
    env.reset(test_days[[10]], rng)
    return env, from_env(env, 0)


def test_centralised_matches_lp(day):
    env, p = day
    c, d = solve_day_lp(env, 0)
    c2, d2 = solve_centralized(p, "efficient_soft")
    assert p.is_household_feasible(c2, d2, tol=1e-5)
    assert p.system_cost(c2, d2) == pytest.approx(p.system_cost(c, d), abs=1e-4)


def test_weak_duality_and_feasible_average(day):
    env, p = day
    c, d = solve_day_lp(env, 0)
    opt = p.system_cost(c, d)
    res = dual_decomposition(p, iters=40, step0=0.02)
    # any dual value is a lower bound, any feasible schedule an upper bound
    assert res.history["dual_bound"][-1] <= opt + 1e-6
    assert p.is_household_feasible(res.c, res.d, tol=1e-6)
    assert p.system_cost(res.c, res.d) >= opt - 1e-6


def test_admm_reaches_the_optimum(day):
    env, p = day
    c, d = solve_day_lp(env, 0)
    opt = p.system_cost(c, d)
    res = sharing_admm(p, rho=0.5, iters=300)
    assert p.is_household_feasible(res.c, res.d, tol=1e-4)
    assert p.system_cost(res.c, res.d) == pytest.approx(opt, rel=0.01)


def test_prox_matches_brute_force():
    rng = np.random.default_rng(1)
    q, h = rng.normal(size=20), rng.normal(size=20)
    N, rho, pen = 6, 0.5, 0.5
    grid = np.linspace(-10, 10, 200001)
    for hard in [False, True]:
        s = prox_g(q, h, N, rho, pen, hard)
        for k in range(20):
            if hard:
                g = np.where(grid <= h[k], 0.0, np.inf)
            else:
                g = pen * np.clip(grid - h[k], 0, None)
            f = g + rho / (2 * N) * (grid - q[k]) ** 2
            assert s[k] == pytest.approx(grid[np.argmin(f)], abs=2e-4)


def test_fairness_orderings(day):
    _, p = day
    out = {}
    for obj in ["efficient", "proportional", "maxmin"]:
        c, d = solve_centralized(p, obj)
        assert p.overload(c - d).sum() < 1e-5  # hard transformer limit
        out[obj] = -(c @ p.a + d @ p.b)
    u = {k: v / p.cap for k, v in out.items()}
    assert out["efficient"].sum() >= out["proportional"].sum() - 1e-4
    assert out["proportional"].sum() >= out["maxmin"].sum() - 1e-4
    assert u["maxmin"].min() >= u["proportional"].min() - 1e-4
    assert u["maxmin"].min() >= u["efficient"].min() - 1e-4


def test_exclusive_milp_never_charges_and_discharges(env, rng):
    _, test_days = data.split_days(len(data.load_days()[0]))
    env.reset(test_days[[12]], rng)  # a day with negative prices
    p = from_env(env, 0)
    c, d = solve_day_lp(env, 0, exclusive=True)
    assert not ((c > 1e-6) & (d > 1e-6)).any()
    cl, dl = solve_day_lp(env, 0)
    assert p.system_cost(c, d) >= p.system_cost(cl, dl) - 1e-6  # the LP is a relaxation
