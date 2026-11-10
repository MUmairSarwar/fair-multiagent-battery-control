"""Local problems solved by each household.

A household only ever sees its own battery, its own load and the signal the
coordinator sends it (a price vector in dual decomposition, a target vector
in ADMM). It never sees the other households' data.
"""

from __future__ import annotations

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from ..data import DT_HOURS
from .problem import DayProblem


def _soc_matrix(T: int, eta: float) -> np.ndarray:
    """Rows give the stored energy after each step minus e0, as a function of [c, d]."""
    L = np.tril(np.ones((T, T)))
    return np.hstack([eta * L, -L / eta])


class LinearHousehold:
    """Household i's LP for a given extra price `lam` on its grid import:

        min  (a + lam) . c + (b - lam) . d   over its own feasible set.

    Used by dual decomposition. The constraint matrix is built once.
    """

    def __init__(self, prob: DayProblem, i: int):
        T = prob.T
        self.prob, self.i, self.T = prob, i, T
        S = _soc_matrix(T, prob.eta)
        e0 = prob.soc_init * prob.cap[i]
        self.A_ub = np.vstack([S, -S])
        self.b_ub = np.concatenate([np.full(T, prob.cap[i] - e0), np.full(T, e0)])
        cmax = prob.pmax[i] * DT_HOURS
        self.bounds = [(0, cmax)] * T + [(0, min(cmax, l)) for l in prob.loads[i]]

    def solve(self, lam: np.ndarray):
        p = self.prob
        cost = np.concatenate([p.a + lam, p.b - lam])
        res = linprog(cost, A_ub=self.A_ub, b_ub=self.b_ub, bounds=self.bounds, method="highs")
        if res.status != 0:
            raise RuntimeError(f"household LP failed: {res.message}")
        c, d = res.x[: self.T], res.x[self.T :]
        return c, d, float(res.fun)


class ProximalHousehold:
    """Household i's ADMM step:

        min  f_i(c, d) + rho/2 * || (c - d) - v ||^2   over its own feasible set,

    where v is the target the coordinator sends. f_i is either its cost
    (alpha = 0, efficiency) or  -cap_i * log(saving_i / cap_i + eps)
    (alpha = 1, proportional fairness on saving per kWh of battery).
    """

    def __init__(self, prob: DayProblem, i: int, rho: float, alpha: int = 0, eps: float = 1e-3):
        T = prob.T
        self.prob, self.i = prob, i
        c, d = cp.Variable(T), cp.Variable(T)
        self.v = cp.Parameter(T)
        cmax = prob.pmax[i] * DT_HOURS
        e = prob.soc_init * prob.cap[i] + cp.cumsum(prob.eta * c - d / prob.eta)
        cons = [c >= 0, c <= cmax, d >= 0, d <= np.minimum(cmax, prob.loads[i]), e >= 0, e <= prob.cap[i]]
        cost = prob.a @ c + prob.b @ d
        if alpha == 0:
            f = cost
            self.solver = cp.OSQP
        elif alpha == 1:
            f = -prob.cap[i] * cp.log(-cost / prob.cap[i] + eps)
            self.solver = cp.CLARABEL
        else:
            raise ValueError("alpha must be 0 or 1")
        obj = f + (rho / 2) * cp.sum_squares(c - d - self.v)
        self.problem = cp.Problem(cp.Minimize(obj), cons)
        self.c, self.d = c, d

    def solve(self, v: np.ndarray):
        self.v.value = v
        if self.solver == cp.OSQP:
            self.problem.solve(solver=cp.OSQP, eps_abs=1e-7, eps_rel=1e-7, max_iter=20000, warm_start=True)
        else:
            # The log objective occasionally trips Clarabel's numerics; SCS is a
            # slower but robust fallback for the same convex problem.
            try:
                self.problem.solve(solver=cp.CLARABEL)
                ok = self.problem.status == cp.OPTIMAL
            except cp.error.SolverError:
                ok = False
            if not ok:
                self.problem.solve(solver=cp.SCS, eps=1e-9, max_iters=200000)
        if self.c.value is None:
            raise RuntimeError(f"household {self.i}: {self.problem.status}")
        return np.asarray(self.c.value), np.asarray(self.d.value)
