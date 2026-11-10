"""ADMM for the sharing problem (Boyd et al. 2011, Section 7.3).

    minimise  sum_i f_i(x_i) + g(sum_i x_i)

x_i = c_i - d_i is household i's extra grid import, f_i is its own cost (plus
the indicator of its battery constraints), and g prices the shared
transformer:

    soft limit:  g(s) = penalty * sum_t max(0, s_t + base_t - limit)
    hard limit:  g(s) = 0 if s_t + base_t <= limit for all t, else +inf

With x_bar the average of the x_i and z_bar the coordinator's copy of it,
one iteration is

    x_i   <- argmin f_i(x_i) + rho/2 || x_i - x_i_old + x_bar - z_bar + u ||^2   (each household)
    z_bar <- argmin g(N z) + N rho/2 || z - u - x_bar ||^2                        (coordinator)
    u     <- u + x_bar - z_bar                                                     (coordinator)

The coordinator's step has a closed form (the proximal operator of a hinge
or of a box), so it is cheap however many households there are. Each
household only receives one vector of 96 numbers per iteration and sends one
back.

With alpha = 1 the household objective is proportional fairness,
f_i = -cap_i * log(saving_i / cap_i + eps); everything else is unchanged.
This is how fairness is put *inside* the distributed algorithm: the
coordinator never needs to know anyone's saving.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .household import ProximalHousehold
from .problem import DayProblem


def prox_g(q: np.ndarray, h: np.ndarray, N: int, rho: float, penalty: float, hard: bool) -> np.ndarray:
    """argmin_s g(s) + rho/(2N) (s - q)^2, elementwise, with g as above and h = limit - base."""
    if hard:
        return np.minimum(q, h)
    shift = penalty * N / rho
    return np.where(q <= h, q, np.where(q - shift >= h, q - shift, h))


@dataclass
class ADMMResult:
    c: np.ndarray
    d: np.ndarray
    iters: int
    history: dict = field(default_factory=dict)


def sharing_admm(
    p: DayProblem,
    rho: float = 2.0,
    iters: int = 300,
    alpha: int = 0,
    hard: bool = False,
    tol: float = 1e-4,
    eps: float = 1e-3,
) -> ADMMResult:
    n, T = p.n, p.T
    homes = [ProximalHousehold(p, i, rho, alpha=alpha, eps=eps) for i in range(n)]
    x = np.zeros((n, T))
    c, d = np.zeros((n, T)), np.zeros((n, T))
    z_bar = np.zeros(T)
    u = np.zeros(T)
    h = p.limit - p.base
    hist = {k: [] for k in ["primal_res", "dual_res", "system_cost", "max_violation_kwh"]}
    k = 0
    for k in range(1, iters + 1):
        x_bar = x.mean(axis=0)
        for i, home in enumerate(homes):
            c[i], d[i] = home.solve(x[i] - x_bar + z_bar - u)
        x = c - d
        x_bar = x.mean(axis=0)
        z_old = z_bar
        z_bar = prox_g(n * (u + x_bar), h, n, rho, p.penalty, hard) / n
        u = u + x_bar - z_bar

        r = np.sqrt(n) * np.linalg.norm(x_bar - z_bar)
        s = rho * np.sqrt(n) * np.linalg.norm(z_bar - z_old)
        hist["primal_res"].append(float(r))
        hist["dual_res"].append(float(s))
        hist["system_cost"].append(p.system_cost(c, d))
        hist["max_violation_kwh"].append(float(np.max(x.sum(axis=0) + p.base - p.limit, initial=0.0)))
        if k > 5 and r < tol and s < tol:
            break
    return ADMMResult(c=c.copy(), d=d.copy(), iters=k, history=hist)
