# Theory notes: distributed and fair scheduling behind a shared transformer

*Part I (Sections 1–6) covers the soft-limit problem behind the results in
[`results/distributed/`](../results/distributed/). Part II (Sections 7–17)
covers exclusive battery modes and the hard-limit extension behind
[`results/distributed/lp_variants/`](../results/distributed/lp_variants/) and
[`results/distributed/hard_limit/`](../results/distributed/hard_limit/).*

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

# Part II. Exclusive battery modes and a hard transformer limit

Part I left two simplifications in place. The LP may charge and discharge the
same battery in the same quarter-hour (Section 5), and the transformer limit
is soft. Part II removes both. Sections 7–11 deal with simultaneous operation
under the original soft limit. Sections 12–16 then switch to a hard limit and
derive the distributed solvers and fairness references used for that model.
The code is in `src/fairbattery/lp_variants.py` (Sections 7–11) and
`src/fairbattery/distributed/hard_limit/` (Sections 12–16); results are in
`results/distributed/lp_variants/` and `results/distributed/hard_limit/`.

The notation follows Part I: price $p_t$ and day-average price $\bar p$ in
EUR/kWh, load $\ell_{i,t}$, capacity $E_i$, one-way efficiency $\eta = 0.95$,
overload penalty $\pi = 0.5$ EUR/kWh.

## 7. Units and the original model

The code measures charge $c_{i,t}$, discharge $d_{i,t}$ and load
$\ell_{i,t}$ in kWh per quarter-hour, not in kW. With $\Delta t = 0.25$ h, a
2 kW charge is $c_{i,t} = 0.5$ kWh, and the 11 kW transformer limit becomes
$h = 2.75$ kWh per step. With stored energy $e_{i,t}$ and grid import
$g_{i,t}$,

$$
e_{i,t+1} = e_{i,t} + \eta c_{i,t} - d_{i,t}/\eta,\qquad
g_{i,t} = \ell_{i,t} + c_{i,t} - d_{i,t},
$$

$$
0 \le c_{i,t} \le \bar c_i,\qquad
0 \le d_{i,t} \le \min(\bar c_i, \ell_{i,t}),\qquad
0 \le e_{i,t} \le E_i,
$$

with $\bar c_i$ the maximum energy per step and $e_{i,0} = E_i/2$. The
original transformer constraint is soft, $\sum_i g_{i,t} \le h + o_t$ with
$o_t \ge 0$, and the cost to minimise is

$$
J(c, d, o) = \sum_{i,t}\Big[p_t (c_{i,t} - d_{i,t}) - \bar p\,(\eta c_{i,t} - d_{i,t}/\eta)\Big] + \pi \sum_t o_t .
$$

The second term values the change in stored energy at the day-average price.
The cost without a battery is a constant and drops out.

## 8. Why the LP charges and discharges at once

Nothing in the LP forces $c_{i,t} d_{i,t} = 0$. Increase both in the same
step by $\Delta c = \delta$ and $\Delta d = \eta^2 \delta$. The stored energy
does not change, $\eta\delta - \eta^2\delta/\eta = 0$, so the valuation term
does not change either, but grid import rises by $(1-\eta^2)\delta$. The
energy cost changes by $p_t(1-\eta^2)\delta$, which is negative when
$p_t < 0$: the battery wastes energy in conversion losses and the household
is paid to import it. At $p_t = 0$ the loop costs nothing, so the solver may
return it as one of several optima.

For $\eta = 0.95$, charging 1 kWh and discharging 0.9025 kWh leaves the
battery as it was and absorbs 0.0975 kWh. At $-0.03$ EUR/kWh that earns
0.002925 EUR. A negative price is necessary for a strict gain but not
sufficient: the loop also needs spare power and, under the soft limit, must
not push the transformer into a costly overload.

On the 51 test days the LP does this in 212 of 29,376 household-steps
(208 at negative prices, 4 at zero), spread over 70 distinct quarter-hours.

## 9. A throughput fee, and when it is enough

Add a fee $\kappa \sum_{i,t}(c_{i,t} + d_{i,t})$, in EUR per kWh of
throughput. The loop above now changes the cost by

$$
\delta\big[p_t (1-\eta^2) + \kappa (1+\eta^2)\big],
$$

so a small fee removes the zero-price ties but not necessarily the
negative-price loops.

**Proposition.** If

$$
\kappa > \max_t \max(0, -p_t)\,\frac{1-\eta^2}{1+\eta^2},
$$

no optimal solution of the LP with fee has $c_{i,t} > 0$ and
$d_{i,t} > 0$ in the same step.

*Proof.* Suppose $c_{i,t}, d_{i,t} > 0$ and let
$\delta = \min(c_{i,t}, d_{i,t}/\eta^2) > 0$. Replace $c_{i,t}$ by
$c_{i,t} - \delta$ and $d_{i,t}$ by $d_{i,t} - \eta^2\delta$. Both stay
non-negative and below their upper bounds, and every $e_{i,t}$ is unchanged.
Grid import falls by $(1-\eta^2)\delta$ but stays non-negative, because
$g_{i,t} = \ell_{i,t} + c_{i,t} - d_{i,t} \ge c_{i,t} \ge 0$ follows from
$d_{i,t} \le \ell_{i,t}$. The transformer constraint only bounds the sum of
imports from above, so it still holds with the same $o_t$. The cost changes
by $-\delta[p_t(1-\eta^2) + \kappa(1+\eta^2)]$, and the bracket is positive:
for $p_t \ge 0$ because $\kappa > 0$, and for $p_t < 0$ by the bound on
$\kappa$. The new point is feasible and strictly cheaper, so the original
point was not optimal. $\square$

The argument uses the structure of this model: upper bounds on import only,
no minimum import, no export and no equality on the grid exchange. With
rooftop PV that must be absorbed, for example, it would need to be redone.
The fee also changes the problem itself, because it makes useful
charge-discharge arbitrage more expensive too. The code uses the bound plus
a margin of $10^{-6}$. Over the 51 test days the most negative price is
about $-31.8$ EUR/MWh, which gives $\kappa \approx 0.00163$ EUR/kWh.

## 10. Exclusive modes as a MILP

The exact fix is one binary $z_{i,t} \in \{0, 1\}$ per household and step:

$$
0 \le c_{i,t} \le \bar c_i\, z_{i,t},\qquad
0 \le d_{i,t} \le \min(\bar c_i, \ell_{i,t})\,(1 - z_{i,t}).
$$

$z = 1$ allows charging, $z = 0$ allows discharging, and idling is possible
in both. The big-M constants are the physical bounds that already exist, so
the formulation is as tight as it can be without extra cuts. Everything else
is unchanged, including the objective without a fee. Since the LP is a
relaxation of this MILP, $J_{\mathrm{LP}} \le J_{\mathrm{MILP}}$, i.e. the
LP's benefit is an upper bound on the MILP's.

The MILP is solved per day with SciPy's HiGHS interface, a relative MIP gap
of $10^{-5}$ and a 60-second time limit. A time-limited or failed solve
raises an error instead of returning an incumbent as if it were optimal;
the largest gap recorded over the 51 days is $9.8 \times 10^{-6}$. A gap of
$10^{-9}$ ran into the time limit on one negative-price day. After the
solve, exclusivity is checked with a tolerance of $10^{-6}$ kWh. The MILP
still uses continuous power, while the RL agents choose from five levels, so
it remains a stronger controller class than the agents.

## 11. How the variants are compared

The original LP, the LP with a small fee (0.0001 EUR/kWh), the LP with the
sufficient fee from Section 9 and the exclusive MILP are solved on the same
51 test days, with the same noisy loads (seed 2025, sampled once), batteries,
soft overload penalty, initial state and end-of-day valuation. Each schedule
is replayed through the environment's accounting. The reported net benefit
never includes the throughput fee; the minimised objective with the fee is
stored separately, so a penalised cost is never compared with an
unpenalised one. Simultaneous operation is counted per household-step,
because several households can overlap in the same quarter-hour.

Results are in the README and in `results/distributed/lp_variants/`; full
schedules are stored as `.npz` files. In short, the exclusive MILP gives up
EUR 0.0017 of benefit over 51 days compared with the LP, so the LP bound used
for the RL comparison is tight to a fraction of a cent.

## 12. The hard-capacity model used for the distributed solvers

From here on the overload variable is dropped and the limit is a constraint,

$$
\sum_i g_{i,t} \le h_t \quad \forall t .
$$

There are two reasons. Zero overload then holds by construction rather than
being an outcome of the penalty. And, as Part I showed, the soft limit
restricts the dual price to $0 \le \lambda \le \pi$, which is easy to get
wrong; with a hard limit the dual feasible set is simply $\lambda \ge 0$.

Write $x_i = (c_i, d_i, e_i) \in \mathbb{R}^{3T}$, let $\mathcal{X}_i$ be the
household's polytope from Section 7, and $g_i(x_i) = \ell_i + c_i - d_i$.
Each day uses its own sufficient fee $\kappa_d$ from that day's minimum
price, so the local cost is linear, $f_i(x_i) = w^\top x_i$ with

$$
w = \big(p - \bar p\,\eta + \kappa_d,\;\; -p + \bar p/\eta + \kappa_d,\;\; 0\big).
$$

The reference problem is

$$
P^\star = \min_{x_i \in \mathcal{X}_i}\; \sum_i f_i(x_i)
\quad\text{s.t.}\quad \sum_i g_i(x_i) \le h .
$$

All numbers in Sections 13–16 refer to this problem and are not comparable
with the soft-limit results of Part I or Sections 7–11. The central reference
is a sparse HiGHS LP with explicit state-of-charge variables, which keeps the
matrix at $O(NT)$ nonzeros instead of the $O(NT^2)$ of the cumulative-sum
formulation in Part I. It agrees with an independent CVXPY/Clarabel model to
within $3 \times 10^{-9}$ EUR on the three test dates (7 January, 14 May and
6 August 2025).

The MILP of Section 10 is not used here. LP duality and the ADMM convergence
theory below do not apply to binary variables. The sufficient fee is what
makes the convex model free of simultaneous operation.

## 13. Dual decomposition with primal recovery

**Dual function.** With $\lambda \in \mathbb{R}^T_{\ge 0}$,

$$
q(\lambda) = \sum_i \min_{x_i \in \mathcal{X}_i}\big[f_i(x_i) + \lambda^\top g_i(x_i)\big] - \lambda^\top h .
$$

Each inner problem is a small LP that needs only the household's own data
and the price vector: $\lambda_t$ is added to the charging cost and
subtracted from the discharging cost. The constant $\lambda^\top \ell_i$ does
not change the minimiser but must be included when $q(\lambda)$ is
evaluated.

**Strong duality.** The problem is a feasible LP (idle batteries fit under
the limit on every scenario used) with bounded variables, so it has a finite
optimum and LP duality gives $\max_{\lambda \ge 0} q(\lambda) = P^\star$.
No Slater condition is needed for this.

**Update.** A supergradient of the concave $q$ at $\lambda^k$ is the
constraint residual $s^k = \sum_i g_i(x_i^k) - h$, which gives

$$
\lambda^{k+1} = \big[\lambda^k + \alpha_k s^k\big]_+,\qquad
\alpha_k = \frac{a}{N k^{0.6}} .
$$

$\lambda$ is in EUR/kWh and $s$ in kWh, so $\alpha_k$ is in EUR/kWh². The
steps satisfy $\sum_k \alpha_k = \infty$ and $\sum_k \alpha_k^2 < \infty$,
the usual conditions for convergence of the dual iterates. A constant step
is included only as a comparison and is not expected to converge exactly. A
Polyak-type step is available as an option; it uses the best feasible cost
found so far instead of $P^\star$, and I make no convergence claim for it.

**Averaging.** The household responses are vertices of $\mathcal{X}_i$ and
jump when $\lambda$ crosses a breakpoint, so the last iterate can overload
the transformer even when $\lambda$ is nearly optimal. The code therefore
also tracks the step-weighted average
$\bar x^k = \sum_{j \le k} \alpha_j x^j / \sum_{j \le k} \alpha_j$, which
stays in each $\mathcal{X}_i$ by convexity, and a second average over the
last half of the iteration budget only, which forgets the early transient.

**Recovery of a feasible schedule.** An average can still violate the limit
slightly after finitely many steps. Each candidate (last iterate and the
averages) is first cleaned of simultaneous operation with the step from the
proof in Section 9, which lowers cost and import. It is then blended with
the idle schedule $x^0$. With idle load $b = \sum_i \ell_i$ and candidate
load $y = \sum_i g_i(\bar x_i)$,

$$
\beta = \min\Big(1,\; \min_{t:\, y_t > b_t} \frac{h_t - b_t}{y_t - b_t}\Big),\qquad
\hat x_i = x_i^0 + \beta\,(\bar x_i - x_i^0).
$$

Since $\sum_i g_i$ is affine in $x$, the blended load is
$b + \beta(y - b) \le h$, and $\hat x_i \in \mathcal{X}_i$ as a convex
combination. The coordinator only needs to broadcast $\beta$. This requires
the idle schedule to be feasible ($b \le h$); the code raises an error
otherwise. That is a restriction of the recovery step, not of the duality
argument.

**Stopping.** The best recovered cost $P_{\text{best}}$ is an upper bound on
$P^\star$ and the best $q(\lambda^k)$ is a lower bound. The algorithm stops
when $P_{\text{best}} - \max_k q(\lambda^k) \le N \cdot \text{tol}$ EUR, so
the stopping rule itself certifies the objective gap. The central optimum is
used only for plots and error reporting.

## 14. Sharing ADMM

**Splitting.** Give each household an allocation $z_i \in \mathbb{R}^T$ and
let $\mathcal{C} = \{z : \sum_i z_{i,t} \le h_t\ \forall t\}$:

$$
\min_{x_i \in \mathcal{X}_i,\, z}\; \sum_i f_i(x_i) + \iota_{\mathcal{C}}(z)
\quad\text{s.t.}\quad g_i(x_i) - z_i = 0 .
$$

This is a two-block problem in the sense of Boyd et al. (2011, §7.3): all
$x_i$ together form one block, which separates by household, and $z$ forms
the other. The scaled-form iteration is

$$
x_i^{k+1} = \arg\min_{x_i \in \mathcal{X}_i}\; f_i(x_i) + \frac{\rho}{2}\big\|g_i(x_i) - z_i^k + u_i^k\big\|_2^2,
$$

a small convex QP per household, solved with a warm-started OSQP model.

**Coordinator step.** With $v_i = g_i(x_i^{k+1}) + u_i^k$, the $z$-update is
the Euclidean projection of $v$ onto $\mathcal{C}$. For each $t$ this is a
projection onto the half-space $\sum_i z_{i,t} \le h_t$, whose normal is the
all-ones vector, so every component is shifted by the same amount:

$$
\delta_t = \Big[\tfrac{1}{N}\Big(\sum_i v_{i,t} - h_t\Big)\Big]_+,\qquad
z_i^{k+1} = v_i - \delta .
$$

$z_i \ge 0$ is not imposed; an allocation may be negative during the
iteration, while the physical imports $g_i$ stay non-negative through the
household constraints. The multiplier update is

$$
u_i^{k+1} = u_i^k + g_i(x_i^{k+1}) - z_i^{k+1} = v_i - z_i^{k+1} = \delta ,
$$

and since $u$ starts at zero, all households always share the same multiplier. The
coordinator receives the aggregate import and broadcasts the single vector
$\delta$, from which each household can form its own $z_i$ and $u_i$. Nobody
receives another household's load or battery parameters. This is a
separation of data, not a privacy guarantee: schedules still reveal
something about loads.

**Residuals.** In the form $Ax + Bz = c$ of Boyd et al., $A$ is block
diagonal with blocks $G = [I, -I, 0]$ and $B = -I$. The primal residual is
$r_i = g_i(x_i) - z_i$ and the dual residual is

$$
s_i = \rho\, G^\top (z_i^{k+1} - z_i^k) = \rho\,(\Delta z_i, -\Delta z_i, 0),
$$

up to sign. Using $\rho\|\Delta z\|$ alone would drop the factor $G^\top$.
The code stops when both RMS values, $\|r\|/\sqrt{NT}$ and
$\|s\|/\sqrt{3NT}$, are below a fixed tolerance. These residuals do not
bound the objective gap directly, so the gap against the central LP is
reported separately, and the final schedule goes through the same cleaning
and idle blending as in Section 13.

**Convergence.** The $f_i$ plus the indicators of $\mathcal{X}_i$, and
$\iota_{\mathcal{C}}$, are closed, proper and convex, and the feasible LP has
a saddle point. Standard two-block ADMM theory (Boyd et al. 2011, §3.2) then
gives convergence of the residuals and the objective for any $\rho > 0$,
assuming exact subproblem solves. OSQP solves them to a finite tolerance,
which is one more reason to check feasibility and the gap numerically. The
optimal schedule need not be unique, so the tests compare objective values
and feasibility rather than arrays.

## 15. Fairness with a hard limit

Let $S_i = -f_i(x_i)$ be household $i$'s daily saving relative to an idle
battery (including the fee and the end-of-day valuation) and
$q_i = S_i / E_i$ its saving per kWh of battery. All fairness models impose
$S_i \ge 0$, so no household is made worse off than with an idle battery.

- **Efficient:** maximise $\sum_i S_i$.
- **Proportional:** maximise $\sum_i \log q_i$. Since
  $\log q_i = \log S_i - \log E_i$, this has the same maximiser as
  $\sum_i \log S_i$. At the optimum $q^\star$,
  $\sum_i (q_i - q_i^\star)/q_i^\star \le 0$ for every feasible positive $q$
  (Kelly et al. 1998).
- **Max-min:** maximise the sorted vector of $q_i$ lexicographically.

The proportional objective here is unweighted and has no offset. Part I
(Section 4) used $\sum_i E_i \log(u_i + \varepsilon)$, a capacity-weighted
variant with $\varepsilon = 10^{-3}$, under the soft limit. The two
objectives are different, so their results are not directly comparable.
Because there is no offset, the code first solves a max-min LP to check that
every household can get a strictly positive saving and raises an error if
not. The log arguments are divided by that achievable floor for numerical
conditioning, which only adds a constant.

Max-min is computed by progressive freezing: maximise a common floor over the
households not yet fixed, find those that cannot exceed it on the optimal
face, fix them at that level (tolerance $2 \times 10^{-7}$ EUR/kWh; a
household counts as blocked if it cannot gain more than ten times that), and
repeat with the rest. A final LP picks the most efficient schedule among the
lexicographically fair ones. This pins down every household's level, not
just the smallest one.

The reported measures are Jain's index, the smallest $q_i$ and the price of
fairness,

$$
J(q) = \frac{(\sum_i q_i)^2}{N \sum_i q_i^2},\qquad
q_{\min} = \min_i q_i,\qquad
\mathrm{PoF} = \frac{\sum_i S_i^{\text{eff}} - \sum_i S_i^{\text{fair}}}{\sum_i S_i^{\text{eff}}} .
$$

$J$ is undefined when all $q_i = 0$ and PoF when the efficient saving is not
positive; the code returns no value in those cases. Proportional fairness
does not guarantee a higher $J$ or a higher $q_{\min}$, and on these three
days it lowers $J$ slightly.

**Distributed proportional fairness.** The sharing ADMM of Section 14 works
unchanged if the local linear cost is replaced by $-\log(S_i/E_i)$; each
household then solves a convex log-plus-quadratic problem (CVXPY with
Clarabel, and SCS as a fallback when Clarabel does not return an optimal
status). Each household scales its log argument by its own best attainable
saving, which again only adds a constant. Max-min is
only computed centrally, as a reference.

## 16. Scaling and message counting

The scaling runs use the same H0 profile and load noise as the rest of the
project. Six households use the original parameters. For $N = 100$ and
$N = 1000$, annual consumption is drawn from the six original levels and
battery size from $\{5, 7.5, 10\}$ kWh (parameter seed 2042, load seed 2025),
and the limit is scaled to $11N/6$ kW. Keeping 11 kW for 1,000 homes would
make the problem infeasible rather than larger. The horizon is always
96 steps, and the scaling date is fixed (7 January 2025).

Runtimes are wall-clock times of a serial simulation, including setting up
the local solvers. They do not measure a real network or parallel speed-up.
Each iteration counts as two messages per household: one up (a $T$-vector
and a cost value) and one down (a $T$-vector plus a few scalars), budgeted
as $2T + 3$ numbers. Total traffic is therefore $2Nk$ messages after $k$
iterations. Every run records whether it met its stopping rule; a run that
hits the iteration limit is reported as not converged, whatever its
objective gap.

## 17. Limitations of Part II

- Perfect day-ahead prices and loads, synthetic H0 households, and a daily
  reset to half-full batteries with end-of-day valuation.
- The throughput fee is a modelling device, not a calibrated degradation
  cost, and the hard-limit results include it.
- The recovery step needs idle batteries to be feasible.
- One aggregate transformer limit with no voltages, reactive power or line
  losses; no export.
- Finite solver tolerances throughout; the MILP is certified only to its
  recorded MIP gap.
- Max-min fairness is a central reference only, and the fairness study covers
  six households on three days.
- Serial timing, synchronous rounds, no delays or lost messages.

## References

- S. Boyd, N. Parikh, E. Chu, B. Peleato, J. Eckstein (2011). Distributed optimization and statistical learning via the alternating direction method of multipliers. *Foundations and Trends in Machine Learning* 3(1).
- D. P. Bertsekas (2016). *Nonlinear Programming*, 3rd ed., Athena Scientific.
- A. Nedić, A. Ozdaglar (2009). Approximate primal solutions and rate analysis for dual subgradient methods. *SIAM J. Optimization* 19(4).
- B. He, X. Yuan (2012). On the O(1/n) convergence rate of the Douglas–Rachford alternating direction method. *SIAM J. Numerical Analysis* 50(2).
- J. Mo, J. Walrand (2000). Fair end-to-end window-based congestion control. *IEEE/ACM Trans. Networking* 8(5).
- F. Kelly, A. Maulloo, D. Tan (1998). Rate control for communication networks: shadow prices, proportional fairness and stability. *J. Operational Research Society* 49.
- D. Bertsimas, V. Farias, N. Trichakis (2011). The price of fairness. *Operations Research* 59(1).
- Solver documentation: [SciPy `milp`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html), [HiGHS](https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/), [OSQP](https://osqp.org/docs/interfaces/python.html), [CVXPY](https://www.cvxpy.org/tutorial/dcp/).
