"""The one-day scheduling problem, written so that it splits by household.

Each household i chooses, for every quarter-hour t,
    c[i, t]  kWh drawn from the grid to charge its battery
    d[i, t]  kWh delivered from its battery to its own house
Its net extra grid import is x[i, t] = c[i, t] - d[i, t].

Household i's own problem only involves its own battery:
    cost_i(c, d) = sum_t a[i, t] c[t] + b[i, t] d[t]       (= minus its saving)
    0 <= c <= cmax,  0 <= d <= min(cmax, load)
    0 <= e0 + cumsum(eta c - d / eta) <= cap

The households are coupled by one thing only, the shared transformer:
    sum_i x[i, t] + base_load[t] <= limit      (kWh per quarter-hour)
Above the limit, every kWh of overload costs `penalty` EUR (soft limit), or
the limit may be treated as hard.

That structure (separable objective + one coupling constraint) is exactly
what dual decomposition and the ADMM "sharing" problem are built for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..data import DT_HOURS


@dataclass
class DayProblem:
    price: np.ndarray  # [T] EUR/kWh
    p_ref: float  # day-average price, values the change in stored energy
    loads: np.ndarray  # [n, T] kWh per quarter-hour
    cap: np.ndarray  # [n] kWh
    pmax: np.ndarray  # [n] kW
    eta: float  # one-way efficiency
    soc_init: float
    limit: float  # transformer limit in kWh per quarter-hour
    penalty: float  # EUR per kWh of overload

    @property
    def n(self) -> int:
        return self.loads.shape[0]

    @property
    def T(self) -> int:
        return self.loads.shape[1]

    @property
    def base(self) -> np.ndarray:
        """Total household load without batteries, [T]."""
        return self.loads.sum(axis=0)

    @property
    def a(self) -> np.ndarray:
        """Cost per kWh of charging, [T]: pay the price, gain stored energy."""
        return self.price - self.p_ref * self.eta

    @property
    def b(self) -> np.ndarray:
        """Cost per kWh of discharging, [T] (negative = it saves money)."""
        return -self.price + self.p_ref / self.eta

    def household_cost(self, c: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Cost of each household (minus its saving), [n]. c, d are [n, T]."""
        return (self.a[None, :] * c + self.b[None, :] * d).sum(axis=1)

    def overload(self, x: np.ndarray) -> np.ndarray:
        """Overload energy per step for net imports x [n, T]."""
        return np.clip(x.sum(axis=0) + self.base - self.limit, 0, None)

    def system_cost(self, c: np.ndarray, d: np.ndarray) -> float:
        """Total cost = minus total saving + overload penalty (the LP objective)."""
        return float(self.household_cost(c, d).sum() + self.penalty * self.overload(c - d).sum())

    def is_household_feasible(self, c: np.ndarray, d: np.ndarray, tol: float = 1e-6) -> bool:
        cmax = (self.pmax * DT_HOURS)[:, None]
        e = self.soc_init * self.cap[:, None] + np.cumsum(self.eta * c - d / self.eta, axis=1)
        return bool(
            (c >= -tol).all()
            and (d >= -tol).all()
            and (c <= cmax + tol).all()
            and (d <= np.minimum(cmax, self.loads) + tol).all()
            and (e >= -tol).all()
            and (e <= self.cap[:, None] + tol).all()
        )


def from_env(env, b: int) -> DayProblem:
    """Day `b` of the current batch of a reset `BatteryCommunity`."""
    return DayProblem(
        price=env.price[b].copy(),
        p_ref=float(env.p_ref[b]),
        loads=env.loads[b].copy(),
        cap=env.cap.copy(),
        pmax=env.pmax.copy(),
        eta=float(env.eta),
        soc_init=float(env.cfg.soc_init),
        limit=float(env.cfg.transformer_kw * DT_HOURS),
        penalty=float(env.cfg.overload_penalty),
    )


def scaled_problem(env, day: int, n_households: int, seed: int = 0) -> DayProblem:
    """A larger neighbourhood for the scaling study.

    The six household types of the main setting are repeated until there are
    `n_households`, each with its own load noise, and the transformer limit is
    scaled in proportion (11 kW per 6 homes). This is synthetic: it tests how
    the algorithms scale, not a real feeder.
    """
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n_households / env.n))
    idx = np.tile(np.arange(env.n), k)[:n_households]
    annual = env.annual[idx]
    s = env.cfg.load_noise_sigma
    base = env.h0_all[day][None, :] * (annual[:, None] / 1000.0)
    loads = base * rng.lognormal(-0.5 * s * s, s, size=base.shape)
    price = env.prices_all[day]
    return DayProblem(
        price=price.copy(),
        p_ref=float(price.mean()),
        loads=loads,
        cap=env.cap[idx].copy(),
        pmax=env.pmax[idx].copy(),
        eta=float(env.eta),
        soc_init=float(env.cfg.soc_init),
        limit=float(env.cfg.transformer_kw * DT_HOURS * n_households / env.n),
        penalty=float(env.cfg.overload_penalty),
    )

