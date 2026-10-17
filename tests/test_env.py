import numpy as np
from pettingzoo.test import parallel_api_test

from fairbattery.env import IDLE_ACTION, BatteryCommunity, EnvConfig, NeighbourhoodParallelEnv


def test_pettingzoo_api(days):
    _, price, h0 = days
    env = NeighbourhoodParallelEnv(EnvConfig(), price, h0)
    parallel_api_test(env, num_cycles=200)


def test_pettingzoo_episode_length(days):
    _, price, h0 = days
    env = NeighbourhoodParallelEnv(EnvConfig(communication=True), price, h0)
    obs, _ = env.reset(seed=1, options={"day": 10})
    assert len(obs) == 6
    steps = 0
    while env.agents:
        obs, rew, term, trunc, info = env.step({a: env.action_space(a).sample() for a in env.agents})
        steps += 1
    assert steps == 96


def test_idle_saves_nothing(env, rng):
    env.reset(np.arange(3), rng)
    done = False
    while not done:
        _, r, done, info = env.step(np.full((3, 6), IDLE_ACTION))
        assert np.allclose(info["saving"], 0)
        assert np.allclose(info["grid_kwh"], info["load_kwh"])


def test_battery_limits_hold(env, rng):
    env.reset(np.arange(4), rng)
    done = False
    while not done:
        a = rng.integers(0, env.n_actions, size=(4, 6))
        e_old = env.energy.copy()
        _, _, done, info = env.step(a)
        assert (env.energy >= -1e-9).all() and (env.energy <= env.cap + 1e-9).all()
        assert (info["grid_kwh"] >= -1e-9).all()  # batteries never export
        assert (info["discharge_kwh"] <= info["load_kwh"] + 1e-9).all()
        assert (info["charge_kwh"] <= env.pmax * 0.25 + 1e-9).all()
        # energy balance of the battery
        expected = e_old + info["charge_kwh"] * env.eta - info["discharge_kwh"] / env.eta
        assert np.allclose(env.energy, expected)


def test_overload_is_measured(days, rng):
    _, price, h0 = days
    env = BatteryCommunity(EnvConfig(transformer_kw=1.0), price, h0)
    env.reset(np.array([0]), rng)
    _, r, _, info = env.step(np.full((1, 6), 4))  # everyone charges at full power
    total = info["grid_kwh"].sum() / 0.25
    assert np.isclose(info["overload_kwh"][0], (total - 1.0) * 0.25)
    assert (r < 0).all()


def test_no_incentive_hides_overload_from_reward(days, rng):
    _, price, h0 = days
    on = BatteryCommunity(EnvConfig(transformer_kw=1.0), price, h0)
    off = BatteryCommunity(EnvConfig(transformer_kw=1.0, price_incentive=False), price, h0)
    on.reset(np.array([0]), np.random.default_rng(5))
    off.reset(np.array([0]), np.random.default_rng(5))
    a = np.full((1, 6), 4)
    _, r_on, _, info = on.step(a)
    _, r_off, _, _ = off.step(a)
    assert np.allclose(r_off, info["saving"] * 10)
    assert r_on.sum() < r_off.sum()
