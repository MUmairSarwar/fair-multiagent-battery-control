"""Neighbourhood battery environment.

Several households sit behind one shared transformer. Each household has a
home battery controlled by its own agent. Every 15 minutes each agent picks a
battery power level. The household then draws

    grid import = household load + battery charging - battery discharging

from the grid and pays the day-ahead price for it. If the sum of all imports
is above the transformer limit, the excess is an overload.

`BatteryCommunity` is the simulator. It is written with numpy and runs a batch
of days side by side, which makes training much faster. `NeighbourhoodParallelEnv`
wraps one day of it in the standard PettingZoo `ParallelEnv` interface.
Both use exactly the same step code.

Accounting (identical for every method, including the LP):
- Saving of household i in a step = what it would have paid without a battery
  minus what it pays now, plus the change in stored energy valued at that
  day's average price. The last term stops a policy from "saving" money by
  simply emptying a battery that started half full.
- Overload energy is charged at `overload_penalty` EUR/kWh. This is a system
  cost. Whether the agents feel it in their reward is the `price_incentive`
  switch.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import DT_HOURS, STEPS_PER_DAY

ACTION_LEVELS = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # fraction of max power, + = charge
IDLE_ACTION = 2


@dataclass
class EnvConfig:
    annual_kwh: list = field(default_factory=lambda: [2500, 3000, 3500, 4000, 4500, 5000])
    capacity_kwh: list = field(default_factory=lambda: [5.0, 10.0, 7.5, 5.0, 10.0, 7.5])
    c_rate: float = 0.5  # max power = c_rate * capacity
    efficiency: float = 0.95  # one-way, so about 90% round trip
    soc_init: float = 0.5
    transformer_kw: float = 11.0
    overload_penalty: float = 0.5  # EUR per kWh above the limit (assumption)
    load_noise_sigma: float = 0.15  # lognormal noise on each quarter-hour of load
    fairness_weight: float = 0.0
    price_incentive: bool = True
    communication: bool = False
    reward_scale: float = 10.0

    @property
    def n_agents(self) -> int:
        return len(self.annual_kwh)


OBS_DIM = 11


class BatteryCommunity:
    """Batched simulator. Arrays are shaped [batch, agents] or [batch, agents, steps]."""

    def __init__(self, cfg: EnvConfig, prices: np.ndarray, h0: np.ndarray):
        self.cfg = cfg
        self.prices_all = prices  # [D, T] EUR/kWh
        self.h0_all = h0  # [D, T] kWh per 1000 kWh/a
        self.n = cfg.n_agents
        self.annual = np.asarray(cfg.annual_kwh, dtype=float)
        self.cap = np.asarray(cfg.capacity_kwh, dtype=float)
        self.pmax = cfg.c_rate * self.cap
        self.eta = cfg.efficiency
        self.n_battery_actions = len(ACTION_LEVELS)
        self.n_actions = self.n_battery_actions * (2 if cfg.communication else 1)
        self.obs_dim = OBS_DIM

    # ------------------------------------------------------------------ setup
    def sample_loads(self, day_idx: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        base = self.h0_all[day_idx][:, None, :] * (self.annual[None, :, None] / 1000.0)
        s = self.cfg.load_noise_sigma
        noise = rng.lognormal(mean=-0.5 * s * s, sigma=s, size=base.shape)
        return base * noise  # kWh per quarter-hour, [B, N, T]

    def reset(self, day_idx, rng: np.random.Generator, loads: np.ndarray | None = None):
        day_idx = np.atleast_1d(np.asarray(day_idx))
        self.day_idx = day_idx
        self.B = len(day_idx)
        self.price = self.prices_all[day_idx]  # [B, T]
        self.loads = self.sample_loads(day_idx, rng) if loads is None else loads
        self.p_ref = self.price.mean(axis=1)  # [B]
        self.t = 0
        self.energy = np.tile(self.cfg.soc_init * self.cap, (self.B, 1))
        self.cum_saving = np.zeros((self.B, self.n))
        self.prev_util = np.zeros(self.B)
        self.prev_bits = np.zeros((self.B, self.n))
        # price features that only use the day-ahead price curve
        p = self.price
        order = p.argsort(axis=1).argsort(axis=1)
        self.price_rank = order / (STEPS_PER_DAY - 1)
        mu = p.mean(axis=1, keepdims=True)
        sd = p.std(axis=1, keepdims=True) + 0.005
        self.price_z = np.clip((p - mu) / sd, -3, 3)
        cheaper_later = np.zeros_like(p)
        for t in range(STEPS_PER_DAY):
            cheaper_later[:, t] = (p[:, t:] < p[:, t : t + 1]).mean(axis=1)
        self.cheaper_later = cheaper_later
        return self.observe()

    # ------------------------------------------------------------ observation
    def observe(self) -> np.ndarray:
        t = min(self.t, STEPS_PER_DAY - 1)
        B, n = self.B, self.n
        obs = np.zeros((B, n, OBS_DIM))
        obs[..., 0] = self.energy / self.cap
        ang = 2 * np.pi * t / STEPS_PER_DAY
        obs[..., 1] = np.sin(ang)
        obs[..., 2] = np.cos(ang)
        obs[..., 3] = self.price_z[:, t, None] / 3
        obs[..., 4] = self.price_rank[:, t, None]
        obs[..., 5] = self.cheaper_later[:, t, None]
        obs[..., 6] = (STEPS_PER_DAY - t) / STEPS_PER_DAY
        mean_load = self.annual / 365 / STEPS_PER_DAY
        obs[..., 7] = self.loads[:, :, t] / mean_load / 3
        obs[..., 8] = self.cap / 10
        if self.cfg.price_incentive:
            obs[..., 9] = self.prev_util[:, None]
        if self.cfg.communication:
            others = self.prev_bits.sum(axis=1, keepdims=True) - self.prev_bits
            obs[..., 10] = others / max(n - 1, 1)
        return obs

    def global_state(self, obs: np.ndarray) -> np.ndarray:
        return obs.reshape(obs.shape[0], -1)

    # ------------------------------------------------------------------- step
    def step(self, actions: np.ndarray):
        """actions: int array [B, N]. Returns obs, reward [B,N], done, info."""
        cfg, t = self.cfg, self.t
        actions = np.asarray(actions)
        level = ACTION_LEVELS[actions % self.n_battery_actions]
        bits = (actions // self.n_battery_actions).astype(float)
        power = level * self.pmax  # kW, + = charge
        load = self.loads[:, :, t]
        price = self.price[:, t]

        e_old = self.energy
        want_charge = np.clip(power, 0, None) * DT_HOURS
        charge = np.minimum(want_charge, (self.cap - e_old) / self.eta)
        want_dis = np.clip(-power, 0, None) * DT_HOURS
        discharge = np.minimum.reduce([want_dis, e_old * self.eta, load])
        e_new = e_old + charge * self.eta - discharge / self.eta
        e_new = np.clip(e_new, 0.0, self.cap)  # floating point only

        grid = load + charge - discharge  # kWh, >= 0 (no export from batteries)
        total_kw = grid.sum(axis=1) / DT_HOURS
        overload_kwh = np.clip(total_kw - cfg.transformer_kw, 0, None) * DT_HOURS

        saving = price[:, None] * (load - grid) + self.p_ref[:, None] * (e_new - e_old)

        reward = saving.copy()
        if cfg.price_incentive:
            tot_charge = charge.sum(axis=1, keepdims=True)
            share = np.where(
                tot_charge > 1e-9,
                charge / np.maximum(tot_charge, 1e-9),
                grid / np.maximum(grid.sum(axis=1, keepdims=True), 1e-9),
            )
            reward -= cfg.overload_penalty * overload_kwh[:, None] * share
        new_cum = self.cum_saving + saving
        if cfg.fairness_weight > 0:
            scale = self.cap.mean()
            f_old = (self.cum_saving / self.cap).std(axis=1)
            f_new = (new_cum / self.cap).std(axis=1)
            reward -= cfg.fairness_weight * scale * (f_new - f_old)[:, None]

        self.energy = e_new
        self.cum_saving = new_cum
        self.prev_util = total_kw / cfg.transformer_kw
        self.prev_bits = bits
        self.t += 1
        done = self.t >= STEPS_PER_DAY
        info = {
            "saving": saving,
            "grid_kwh": grid,
            "load_kwh": load,
            "total_kw": total_kw,
            "overload_kwh": overload_kwh,
            "charge_kwh": charge,
            "discharge_kwh": discharge,
        }
        return self.observe(), reward * cfg.reward_scale, done, info


# ---------------------------------------------------------------- PettingZoo
try:
    from gymnasium import spaces
    from pettingzoo import ParallelEnv
except ImportError:  # pragma: no cover
    ParallelEnv = object


class NeighbourhoodParallelEnv(ParallelEnv):
    """PettingZoo wrapper: one episode = one day of 96 quarter-hours."""

    metadata = {"name": "fair_battery_neighbourhood_v0", "render_modes": []}

    def __init__(self, cfg: EnvConfig, prices: np.ndarray, h0: np.ndarray, days=None):
        self.core = BatteryCommunity(cfg, prices, h0)
        self.days = np.arange(len(prices)) if days is None else np.asarray(days)
        self.possible_agents = [f"house_{i}" for i in range(cfg.n_agents)]
        self.agents = []
        self._rng = np.random.default_rng()
        self._obs_space = spaces.Box(-2.0, 2.0, shape=(OBS_DIM,), dtype=np.float32)
        self._act_space = spaces.Discrete(self.core.n_actions)

    def observation_space(self, agent):
        return self._obs_space

    def action_space(self, agent):
        return self._act_space

    def _pack(self, obs):
        return {a: obs[0, i].astype(np.float32) for i, a in enumerate(self.agents)}

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        if options and "day" in options:
            day = int(options["day"])
        else:
            day = int(self._rng.choice(self.days))
        self.agents = list(self.possible_agents)
        obs = self.core.reset(np.array([day]), self._rng)
        infos = {a: {"day": day} for a in self.agents}
        return self._pack(obs), infos

    def step(self, actions):
        act = np.array([[actions[a] for a in self.possible_agents]])
        obs, reward, done, info = self.core.step(act)
        observations = self._pack(obs)
        rewards = {a: float(reward[0, i]) for i, a in enumerate(self.agents)}
        terminations = {a: bool(done) for a in self.agents}
        truncations = {a: False for a in self.agents}
        infos = {
            a: {
                "saving_eur": float(info["saving"][0, i]),
                "grid_kwh": float(info["grid_kwh"][0, i]),
                "total_kw": float(info["total_kw"][0]),
                "overload_kwh": float(info["overload_kwh"][0]),
            }
            for i, a in enumerate(self.agents)
        }
        if done:
            self.agents = []
        return observations, rewards, terminations, truncations, infos
