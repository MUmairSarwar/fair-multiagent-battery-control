"""Variants of the daily soft-limit LP used to study simultaneous charge and discharge.

`solve_day` solves one day as the original LP, as the LP with a throughput
fee, or as the exclusive-mode MILP. `run_variant` replays a variant through
the environment's accounting for a whole batch. The derivations are in
docs/theory.md, Sections 7-11.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds, LinearConstraint, linprog, milp

from .baselines import build_day_lp
from .data import DT_HOURS, STEPS_PER_DAY
from .env import BatteryCommunity


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
    """Fee per kWh of throughput that rules out simultaneous charge and discharge.

    Returns the bound from theory Section 9 plus ``margin``. It holds for this
    model only (upper limits on import, no minimum import). The fee changes
    the objective, so the result is not the MILP optimum.
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
    """Solve one day as an LP or MILP; raise if optimality is not proven.

    c and d are energies per step (kWh), not power. The soft transformer
    penalty and the end-of-day valuation are the same as in the original LP.
    """
    if not np.isfinite(throughput_cost) or throughput_cost < 0:
        raise ValueError('throughput_cost must be finite and nonnegative')
    if not 0 < env.eta <= 1 or env.cfg.overload_penalty < 0:
        raise ValueError('invalid efficiency or overload penalty')
    if not 0 <= b < env.B:
        raise IndexError('day index outside current batch')
    original, A, rhs, bound_pairs = build_day_lp(env, b)
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


def run_variant(env: BatteryCommunity, *, throughput_cost: float = 0.0, exclusive: bool = False):
    """Solve each day and replay the original environment accounting.

    By default this is the original relaxed LP, so the published numbers can
    be reproduced. ``exclusive=True`` solves the exclusive-mode MILP. Savings
    never include the throughput fee; the fee and the minimised objective are
    returned separately.
    """
    B, n, T = env.B, env.n, STEPS_PER_DAY
    charge = np.zeros((B, n, T))
    discharge = np.zeros((B, n, T))
    solutions = []
    for b in range(B):
        result = solve_day(env, b, throughput_cost=throughput_cost, exclusive=exclusive)
        charge[b], discharge[b] = result.charge, result.discharge
        solutions.append(result)
    eta = env.eta
    e0 = env.cfg.soc_init * env.cap
    flow = charge * eta - discharge / eta
    energy = e0[None, :, None] + np.cumsum(flow, axis=2)
    e_prev = np.concatenate([np.repeat(e0[None, :, None], B, 0), energy[:, :, :-1]], axis=2)
    grid = env.loads + charge - discharge
    total_kw = grid.sum(axis=1) / DT_HOURS
    overload = np.clip(total_kw - env.cfg.transformer_kw, 0, None) * DT_HOURS
    saving = env.price[:, None, :] * (env.loads - grid) + env.p_ref[:, None, None] * (energy - e_prev)
    both = ((charge > 1e-6) & (discharge > 1e-6)).sum()
    return {
        "saving": saving,
        "charge_kwh": charge,
        "discharge_kwh": discharge,
        "energy_kwh": energy,
        "objective_eur": np.array([r.objective for r in solutions]),
        "throughput_fee_eur": np.array([r.throughput_fee for r in solutions]),
        "solver_seconds": np.array([r.solve_seconds for r in solutions]),
        "mip_gaps": np.array([r.mip_gap for r in solutions]),
        "total_kw": total_kw,
        "overload_kwh": overload,
        "grid_kwh": grid,
        "simultaneous_charge_discharge_steps": int(both),
    }


def run_milp(env: BatteryCommunity):
    """Evaluate the exclusive-mode baseline using the unchanged economic score."""
    return run_variant(env, exclusive=True)


def run_sufficient_fee_lp(env: BatteryCommunity):
    """LP with the sufficient throughput fee, so no battery charges and discharges at once.

    One fee is computed from the lowest price in the whole batch. This
    changes the objective, so the result is not the MILP optimum; the
    original score is reported separately.
    """
    fee = sufficient_throughput_cost(env.price, env.eta)
    result = run_variant(env, throughput_cost=fee)
    if result['simultaneous_charge_discharge_steps']:
        raise RuntimeError('Sufficient-fee LP still charges and discharges at once')
    result['throughput_cost_eur_per_kwh'] = fee
    return result
