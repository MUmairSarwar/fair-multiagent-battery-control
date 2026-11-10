# Theory notes: distributed and fair scheduling behind a shared transformer

*Written with AI assistance as part of this learning project. The results it
refers to are in [`results/distributed/`](../results/distributed/).*

## 1. The problem and its structure

For one day with $T = 96$ quarter-hours and $n$ households, household $i$
chooses $c_i, d_i \in \mathbb{R}^T$ (kWh charged from the grid and kWh
discharged into the house). Its own feasible set $\mathcal{X}_i$ is

$$
0 \le c_i \le \bar c_i,\quad 0 \le d_i \le \min(\bar c_i, \ell_i),\quad
0 \le e_i^0 + \textstyle\sum_{k \le t} (\eta c_{i,k} - d_{i,k}/\eta) \le E_i \;\; \forall t,
$$

a polytope, and its cost (minus its saving) is linear,
$f_i(c_i, d_i) = a^\top c_i + b^\top d_i$ with
$a = p - \bar p\,\eta$ and $b = -p + \bar p/\eta$
($p$ the price, $\bar p$ the day-average price that values stored energy).

The households are coupled only through the transformer. With
$x_i = c_i - d_i$ and base load $L = \sum_i \ell_i$,

$$
\min_{(c_i,d_i)\in\mathcal{X}_i,\; o \ge 0}\;
\sum_i f_i(c_i,d_i) + \pi \mathbf{1}^\top o
\quad\text{s.t.}\quad \sum_i x_i + L - o \le h,
$$

where $h$ is the limit and $\pi$ the overload penalty. A separable objective
with one coupling constraint is the textbook setting for both dual
decomposition and ADMM.

## 2. Dual decomposition

**Lagrangian.** Attach $\lambda \in \mathbb{R}^T_{\ge 0}$ to the coupling
constraint:

$$
\mathcal{L} = \sum_i \big[(a+\lambda)^\top c_i + (b-\lambda)^\top d_i\big]
+ (\pi\mathbf{1} - \lambda)^\top o + \lambda^\top (L - h).
$$

Minimising over $o \ge 0$ gives $0$ if $\lambda \le \pi$ and $-\infty$
otherwise, so the dual feasible set is $0 \le \lambda \le \pi$ (with a hard
limit there is no $o$ and only $\lambda \ge 0$). The dual function separates:

$$
g(\lambda) = \sum_i \min_{(c_i,d_i)\in\mathcal{X}_i}
\big[(a+\lambda)^\top c_i + (b-\lambda)^\top d_i\big] + \lambda^\top (L-h).
$$

**Interpretation.** $\lambda_t$ is a congestion price in EUR/kWh added to the
energy price in quarter-hour $t$. Each household reacts to it alone; nobody
shares their load or battery data.

**Bounds.** By weak duality every $g(\lambda)$ is a lower bound on the optimal
cost. The primal is a feasible, bounded LP (doing nothing is feasible), so
strong duality holds and $\max g = $ optimum.

**Algorithm.** $g$ is concave and piecewise linear. A supergradient at
$\lambda$ is the constraint residual $r = \sum_i x_i(\lambda) + L - h - o$,
which gives projected supergradient ascent

$$
\lambda^{k+1} = \Pi_{[0,\pi]}\big(\lambda^k + \alpha_k r^k\big),\qquad
\alpha_k = \alpha_0/\sqrt{k}.
$$

With diminishing, non-summable steps the best dual value converges to the
optimum (Bertsekas, *Nonlinear Programming*, Ch. 6).

**Why the schedule oscillates, and the fix.** Each household solves an LP, so
its response is a vertex of $\mathcal{X}_i$ and jumps as $\lambda$ crosses a
breakpoint. The last iterate can therefore overload the transformer even when
$\lambda$ is nearly optimal. The running average
$\bar x^k = \frac1k\sum_{j\le k} x^j$ lies in each $\mathcal{X}_i$ (convex
sets), and its constraint violation goes to zero (Nedić & Ozdaglar, 2009).
That average is what the experiments report.

## 3. ADMM for the sharing problem

Write the problem as $\min \sum_i F_i(x_i) + G(\sum_i x_i)$, where
$F_i$ is the household cost plus the indicator of its feasible set and

$$
G(s) = \pi \textstyle\sum_t \max(0, s_t + L_t - h)\quad\text{(soft)},
\qquad G(s) = \iota_{\{s + L \le h\}}(s)\quad\text{(hard)}.
$$

Following Boyd et al. (2011, §7.3), introduce copies $z_i$ of $x_i$ and the
average $\bar z$. Scaled ADMM then reduces to

$$
\begin{aligned}
x_i^{k+1} &= \arg\min_{x_i}\; F_i(x_i) + \tfrac{\rho}{2}\big\|x_i - x_i^k + \bar x^k - \bar z^k + u^k\big\|^2,\\
\bar z^{k+1} &= \arg\min_{\bar z}\; G(n\bar z) + \tfrac{n\rho}{2}\big\|\bar z - u^k - \bar x^{k+1}\big\|^2,\\
u^{k+1} &= u^k + \bar x^{k+1} - \bar z^{k+1}.
\end{aligned}
$$

Each household solves a small QP with its own data. The coordinator only
handles $T$ numbers.

**The coordinator step in closed form.** With $s = n\bar z$,
$q = n(u + \bar x)$ and $\theta = h - L$, the step is, for each $t$,
$\min_s \pi\max(0, s-\theta) + \frac{\rho}{2n}(s-q)^2$. Checking the three
cases (smooth part left of the kink, at the kink, right of the kink) gives

$$
s^\star = \begin{cases} q & q \le \theta \\ q - \pi n/\rho & q - \pi n/\rho \ge \theta \\ \theta & \text{otherwise}\end{cases}
\qquad\text{(soft)},\qquad s^\star = \min(q, \theta)\quad\text{(hard)}.
$$

`tests/test_distributed.py` checks this against a brute-force grid search.

**Convergence.** If every $F_i$ and $G$ is closed, proper and convex and the
unaugmented Lagrangian has a saddle point, then the ADMM residuals go to zero
and the objective converges to the optimum for any $\rho > 0$ (Boyd et al.
2011, §3.2). All of those hold here: polyhedral sets, linear or concave-log
objectives, and a hinge or box penalty. The worst-case ergodic rate is
$O(1/k)$ (He & Yuan, 2012); in practice $\rho$ matters a lot (see the tuning
note in `scripts/run_distributed.py`).

**Stopping.** Primal residual $\sqrt n\,\|\bar x - \bar z\|$ and dual residual
$\rho\sqrt n\,\|\bar z^{k+1} - \bar z^k\|$, both below $10^{-4}$, or 300
iterations.

## 4. Fairness

Let $u_i = s_i / E_i$ be household $i$'s saving per kWh of battery, so
households with different battery sizes are compared fairly. With weights
$E_i$, the family of $\alpha$-fair objectives (Mo & Walrand, 2000) is

$$
\max \sum_i E_i\, U_\alpha(u_i),\qquad
U_0(u) = u\;(\text{efficient}),\quad U_1(u) = \log u\;(\text{proportional, Kelly 1998}),\quad
\alpha\to\infty:\; \max\min_i u_i\;(\text{max-min}).
$$

The log needs $u_i > 0$, so $\log(u_i + \varepsilon)$ with
$\varepsilon = 10^{-3}$ EUR/kWh is used. Doing nothing gives $u_i = 0$, so it
is always feasible.

**Fairness inside the distributed algorithm.** Proportional fairness keeps
the objective separable, $\sum_i -E_i\log(u_i(x_i)+\varepsilon)$, and convex
(the negative log of an affine function). So it fits ADMM unchanged: only the
households' local objectives change, and the coordinator still never sees
anyone's saving. Max-min fairness couples everyone through the minimum. It
needs a shared epigraph variable, so here it is solved centrally as a
reference.

**Price of fairness.** The relative loss of total saving compared with the
efficient allocation (Bertsimas, Farias & Trichakis, 2011).

## 5. The LP relaxation artefact

The published LP sometimes charges and discharges one battery in the same
quarter-hour. That happens when the battery is already full and the price is
zero or negative. With $d = \eta^2 c$, stored energy stays the same while grid
import rises by $c(1-\eta^2)$, which the household is *paid* to consume when
$p < 0$. The conversion losses absorb the energy. A real inverter cannot do
both at once, so the LP is a relaxation: still a valid upper bound, slightly
loose. One binary per household and step removes it (`exclusive=True`, a
MILP). Over the 51 test days the difference is a fraction of a cent (see
`results/distributed/summary.json`).

## 6. Limitations

- Perfect foresight of prices and loads; no forecast errors yet.
- One transformer limit, no network model (no voltages or line flows).
- The step size $\alpha_0$ and penalty $\rho$ were tuned on two days, not
  derived.
- The scaling study uses synthetic copies of the six household types.
- Synchronous rounds, no communication delays or dropped messages.

## References

- S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning* 3(1).
- D. P. Bertsekas (2016). *Nonlinear Programming*, 3rd ed., Athena Scientific.
- A. Nedić, A. Ozdaglar (2009). Approximate primal solutions and rate analysis for dual subgradient methods. *SIAM J. Optimization* 19(4).
- B. He, X. Yuan (2012). On the O(1/n) convergence rate of the Douglas–Rachford alternating direction method. *SIAM J. Numerical Analysis* 50(2).
- J. Mo, J. Walrand (2000). Fair end-to-end window-based congestion control. *IEEE/ACM Trans. Networking* 8(5).
- F. Kelly, A. Maulloo, D. Tan (1998). Rate control for communication networks: shadow prices, proportional fairness and stability. *J. Operational Research Society* 49.
- D. Bertsimas, V. Farias, N. Trichakis (2011). The price of fairness. *Operations Research* 59(1).
