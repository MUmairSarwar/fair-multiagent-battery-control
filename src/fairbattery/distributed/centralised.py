"""Auditable LP relaxation and exclusive-mode MILP, with identical accounting."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from fairbattery.baselines import _build_day_lp
from fairbattery.data import DT_HOURS, STEPS_PER_DAY


@dataclass(frozen=True)
class DaySolution:
    charge: np.ndarray
    discharge: np.ndarray
    objective: float  # minimised cost INCLUDING any throughput fee
    original_objective: float  # negative original environment net benefit
    throughput_fee: float
    solve_seconds: float
    mip_gap: float  # NaN for LP


def sufficient_throughput_cost(prices, eta: float, margin: float = 1e-6) -> float:
    """Strict sufficient fee for removing cycles in this particular model.

    Assumes upper-only transformer constraints and no lower import requirement.
    This changes the economic objective; it is not an exact MILP reformulation.
    """
    prices = np.asarray(prices, dtype=float)
    if prices.size == 0 or not np.isfinite(prices).all():
        raise ValueError('prices must be nonempty and finite')
    if not 0 < eta <= 1 or not np.isfinite(margin) or margin <= 0:
        raise ValueError('eta must be in (0, 1] and margin must be positive')
    threshold = max(0.0, -float(prices.min())) * (1 - eta**2) / (1 + eta**2)
    return threshold + margin


def solve_day(env, b: int, *, throughput_cost: float = 0.0,
              exclusive: bool = False, time_limit: float = 60.0,
              mip_rel_gap: float = 1e-5) -> DaySolution:
    """Solve one day; raise rather than silently accept a time-limited incumbent.

    c,d are grid-side ENERGY per step (kWh), not power. The existing soft
    transformer penalty and terminal energy valuation are preserved exactly.
    """
    if not np.isfinite(throughput_cost) or throughput_cost < 0:
        raise ValueError('throughput_cost must be finite and nonnegative')
    if not 0 < env.eta <= 1 or env.cfg.overload_penalty < 0:
        raise ValueError('invalid efficiency or overload penalty')
    if not 0 <= b < env.B:
        raise IndexError('day index outside current batch')
    original, A, rhs, bound_pairs = _build_day_lp(env, b)
    n, T = env.n, STEPS_PER_DAY
    m = n * T
    cost = original.copy()
    cost[:2*m] += throughput_cost
    lower = np.array([lo for lo, hi in bound_pairs])
    upper = np.array([np.inf if hi is None else hi for lo, hi in bound_pairs])
    start = perf_counter()
    if exclusive:
        # z_it = 1 permits charging; z_it = 0 permits discharging.
        # Tight M values are the pre-existing physical per-step upper bounds.
        mc, md = upper[:m], upper[m:2*m]
        # Explicit SOC variables avoid the dense cumulative-state rows of the
        # legacy LP. Order: c, d, overload, e_after, binary_mode.
        I = sp.eye(m, format='csr')
        Z = sp.csr_matrix((m, m))
        O = sp.csr_matrix((m, T))
        previous = sp.diags(np.ones(T-1), -1, shape=(T,T), format='csr')
        previous = sp.kron(sp.eye(n), previous, format='csr')
        dynamics = sp.hstack([env.eta*I, -I/env.eta, O, previous-I, Z], format='csr')
        initial = np.zeros((n,T))
        initial[:,0] = -env.cfg.soc_init*env.cap
        transformer = sp.hstack([A[-T:], sp.csr_matrix((T,2*m))], format='csr')
        charge_mode = sp.hstack([I, Z, O, Z, -sp.diags(mc)], format='csr')
        discharge_mode = sp.hstack([Z, I, O, Z, sp.diags(md)], format='csr')
        combined = sp.vstack([transformer, dynamics, charge_mode, discharge_mode], format='csr')
        rhs_upper = np.r_[rhs[-T:], initial.ravel(), np.zeros(m), md]
        rhs_lower = np.r_[np.full(T,-np.inf), initial.ravel(), np.full(2*m,-np.inf)]
        res = milp(
            np.r_[cost, np.zeros(2*m)],
            integrality=np.r_[np.zeros(len(cost)+m), np.ones(m)],
            bounds=Bounds(np.r_[lower, np.zeros(2*m)],
                          np.r_[upper, np.repeat(env.cap,T), np.ones(m)]),
            constraints=LinearConstraint(combined, rhs_lower, rhs_upper),
            options={'time_limit': time_limit, 'mip_rel_gap': mip_rel_gap},
        )
    else:
        res = linprog(cost, A_ub=A, b_ub=rhs, bounds=bound_pairs,
                      method='highs', options={'time_limit': time_limit})
    elapsed = perf_counter() - start
    if not res.success or res.x is None:
        raise RuntimeError(f'{"MILP" if exclusive else "LP"} failed: {res.message}; gap={getattr(res, "mip_gap", None)}; objective={getattr(res, "fun", None)}')
    x = res.x[:len(cost)]
    c, d = x[:m].reshape(n, T), x[m:2*m].reshape(n, T)
    if exclusive and np.any((c > 1e-6) & (d > 1e-6)):
        raise RuntimeError('MILP solution violates exclusive modes at 1e-6 kWh tolerance')
    return DaySolution(c, d, float(cost @ x), float(original @ x),
                       float(throughput_cost * (c.sum() + d.sum())),
                       elapsed, float(getattr(res, 'mip_gap', np.nan)))
