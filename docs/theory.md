# Theory notes: distributed and fair scheduling behind a shared transformer

*The results these notes refer to are in
[`results/distributed/`](../results/distributed/).*

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


---

# Distributed optimization: model and derivations

**Authorship status:** AI-assisted learning draft, not yet an independently
written author derivation. Muhammad Umair Sarwar should verify each argument
and rewrite it in his own words before describing the derivations as authored
by him. The implementation was built with AI assistance.

Stages 0–4 are implemented. The sections follow the learning sequence, with
explicit distinctions between the historical soft limit and the new hard limit.

## 1. What the existing variables mean

Let i denote a household and t a quarter-hour. The existing code measures
charge c_it, discharge d_it and load l_it in **kWh per interval**, not kW.
The interval length is Delta t = 0.25 hours. Thus a 2 kW charge over one
interval is c_it = 0.5 kWh. The shared 11 kW limit becomes 2.75 kWh per interval.

With one-way efficiency eta = 0.95 and stored energy e_it:

\[
e_{i,t+1}=e_{it}+\eta c_{it}-d_{it}/\eta,
\qquad g_{it}=l_{it}+c_{it}-d_{it}.
\]

The restrictions are

\[
0\leq c_{it}\leq P_i\Delta t,\quad
0\leq d_{it}\leq\min(P_i\Delta t,l_{it}),\quad
0\leq e_{it}\leq B_i.
\]

Each day starts at e_i0 = 0.5 B_i. There is no fixed terminal SOC; the final
energy change is valued at that day's mean price p_ref. Prices pi_t are
EUR/kWh. These conventions are preserved in Stage 0.

The existing transformer constraint is **soft**:

\[
\sum_i g_{it}\leq 11\Delta t+o_t,\qquad o_t\geq0.
\]

Overload costs gamma o_t, with gamma = 0.5 EUR/kWh. The optimizer is allowed
to overload if that is worth the penalty. Zero measured overload is therefore
an outcome, not a hard feasibility guarantee. We will explicitly choose soft
versus hard capacity explicitly; Section 8 makes that choice for Stage 1.

## 2. The exact original objective

The minimization objective (negative scored net benefit) is

\[
J(c,d,o)=\sum_{it}\left[
\pi_t(c_{it}-d_{it})-p_{\rm ref}(\eta c_{it}-d_{it}/\eta)
\right]+\gamma\sum_t o_t.
\]

The no-battery load cost is a constant and cancels when computing savings.
Summing the p_ref term over time gives terminal energy minus initial energy.
This is not the same as imposing a cyclic terminal state.

## 3. Why the relaxed LP cycles

The LP never imposes c_it d_it = 0. Consequently, both variables may be
positive. This is a model relaxation, not an error in HiGHS.

Consider an SOC-neutral loop:

\[
\Delta c=\delta,\qquad \Delta d=\eta^2\delta.
\]

Then Delta e = eta delta - eta^2 delta / eta = 0, but net import rises by
(1 - eta^2) delta. The terminal valuation cancels. At a negative price,
pi_t (1 - eta^2) delta < 0: wasting energy reduces the objective, provided
there is enough capacity and the other bounds permit the loop. At zero
price, the loop can be cost-neutral, so the LP can return a cycling solution
among several optima. Binding constraints/overload penalties can prevent or
reduce this incentive; negative price alone is not a sufficient condition
for cycling in every interval.

For eta = 0.95, charging 1 kWh and discharging 0.9025 kWh leaves SOC unchanged
but consumes 0.0975 kWh. At -0.03 EUR/kWh this earns 0.002925 EUR before any
incremental overload cost. It is economically attractive in the relaxed
model but is not an allowed single-mode battery action in the simulator.

## 4. Linear throughput penalty: benefit and limitation

Add kappa sum_it(c_it+d_it) to J, where kappa is EUR per AC-side kWh of
throughput. The same loop now costs

\[
\delta[\pi_t(1-\eta^2)+\kappa(1+\eta^2)].
\]

A small positive fee eliminates cost-neutral cycling but may not remove
profitable negative-price cycling. For this model, a sufficient bound is

\[
\kappa>\max_t\max(0,-\pi_t)\frac{1-\eta^2}{1+\eta^2}.
\]

**Why sufficient here:** if c,d > 0, remove delta = min(c,d/eta^2) from c
and eta^2 delta from d. All SOC values remain unchanged. Upper charging and
discharging bounds remain satisfied; grid import decreases but stays
nonnegative because d <= load still holds. Upper-only transformer constraints
become easier. Keep overload slack unchanged, or decrease it. Under the
strict bound above, the objective strictly improves. Thus an optimum cannot
contain such a loop (up to solver tolerances).

This proof is specific to these constraints. It does not automatically hold
with lower import limits, must-take PV or other equalities on grid exchange.
A sufficient penalty **changes the optimization problem**, including legitimate
arbitrage. It does not necessarily reproduce the unpenalized physical optimum.

## 5. Exact mode restriction: MILP

For every household-step introduce z_it in {0,1}:

\[
0\leq c_{it}\leq(P_i\Delta t)z_{it},\qquad
0\leq d_{it}\leq\min(P_i\Delta t,l_{it})(1-z_{it}).
\]

If z = 1, discharge is zero. If z = 0, charge is zero. Idling is feasible in
either mode. These use the tight existing physical bounds, not an arbitrary
large big-M. The remaining objective and constraints are unchanged.

We solve this mixed-integer linear program with SciPy's HiGHS interface.
The requested relative MIP gap is 1e-5 per day. The code raises on a solver
failure or time limit, rather than labelling an unproven incumbent optimal.
It also checks exclusivity at 1e-6 kWh tolerance. See the recorded max MIP
gap for the numerical certification of each experiment.

For the same minimization objective, J_LP <= J_MILP. Equivalently, the relaxed
LP's maximum economic benefit is an upper bound on the exclusive model's
benefit. The MILP still permits continuous power, whereas MARL uses five
power levels, so it remains a stronger controller class than MARL.

## 6. How the comparison is kept fair

- Exactly the same 51 held-out days, seed 2025 and noisy H0 loads as the
  repository's original test evaluation; loads sampled once before all solves.
- Identical efficiency, batteries, soft overload penalty, initial SOC and
  terminal valuation for every variant.
- Original LP; small-fee LP (0.0001 EUR/kWh); sufficient-fee LP (bound + 1e-6);
  and, optionally, exclusive MILP without a throughput fee.
- Record original economic benefit separately from the optimized penalized
  objective. Never compare a penalized cost directly with an unpenalized cost.
- Count household-steps separately from clock steps. One interval may have
  simultaneous operation in multiple households.
- Tests cover real-data battery feasibility, objective accounting, a deliberately
  adverse negative-price case, failure of a tiny fee, and success of the MILP.

The default reproducible benchmark uses the three LPs. Add `--include-milp`
to attempt the optional binary baseline; a time-limited run raises rather than
reporting an unproven optimum. The default per-day time limit is 60 seconds. A 1e-9 requested MIP gap hit
a time limit on an affected negative-price day; the reported binary benchmark
uses the explicitly documented 1e-5 relative tolerance instead. This is a
numerical optimality tolerance, not a proof of exact arithmetic optimality.

`results/distributed/stage0/comparison.csv` and `comparison.json` contain
actual run results. `.npz` files preserve full schedules. Figures are generated
by `scripts/audit_stage0.py`, not hand-edited.

## 7. Boundary before Stage 1

The exclusive MILP is a physical reference. Standard LP strong duality and
convex ADMM convergence guarantees do not transfer to its binary variables.
For later convex distributed experiments we must choose a clearly labelled
LP relaxation or a throughput-penalized LP, then compare it with the same
centralized formulation. If retaining the soft transformer model, its overload
slack also changes the dual price domain. Sections 8–13 implement and explain the hard-capacity, throughput-penalized choice.

## Check your understanding

1. Why does the 11 kW limit become 2.75 kWh per step?
2. With eta=0.95, why can a loop consume electricity without changing SOC?
3. Why is a tiny positive throughput fee insufficient at some negative prices?
4. Why can a MILP have lower benefit than the relaxed LP without either solver
   being wrong?
5. Why must we distinguish a hard transformer limit from a penalty for exceeding it?

## References

- SciPy MILP API: https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.milp.html
- SciPy HiGHS LP API: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-highs.html
- Original model: `src/fairbattery/baselines.py` and `src/fairbattery/env.py`.

The algebraic loop-removal proof above is a derivation for this repository's
specific model, not a claim that every battery LP relaxation is exact.

# Stages 1–4: completed convex distributed extension

These sections remain an **AI-assisted derivation draft for author review**.
The implementation does not make an unsupported claim of independent human
authorship. Stage 0's historical experiments are retained above.

## 8. A deliberate model choice: hard capacity

For these stages only, remove overload slack and impose

\[
\sum_i g_i(x_i)\le K,\qquad K_t=P_{\rm transformer}\Delta t.
\]

Here x_i = (c_i,d_i,e_i), X_i is household i's local battery polytope,
g_i(x_i)=l_i+c_i-d_i, and f_i(x_i)=a_i^T x_i is the original negative savings
**plus the sufficient throughput fee**. Every day uses its own sufficient
fee, computed from that day's minimum price. The shared cost vector is

\[
a_i=(\pi-p_{\rm ref}\eta+\kappa,
      -\pi+p_{\rm ref}/\eta+\kappa,0).
\]

The centralized reference is

\[
P^*=\min_{x_i\in X_i}\sum_i f_i(x_i)
\quad\text{subject to}\quad\sum_i g_i(x_i)\le K.
\]

All numerical comparisons within Stages 1–4 use **this same problem**. Their
numbers are not interchangeable with the unpenalized soft-limit Stage 0 table.
SciPy/HiGHS supplies a sparse central LP, independently checked against a
CVXPY/Clarabel formulation on three dates. The battery states are explicit,
so the sparse model has O(NT) nonzeros rather than cumulative O(NT²) rows.

## 9. Lagrangian and dual decomposition

For a minimization problem with an upper inequality, nonnegative multipliers
add a congestion charge:

\[
L(x,\lambda)=\sum_i[f_i(x_i)+\lambda^T g_i(x_i)]-\lambda^TK,
\qquad\lambda\ge0.
\]

Thus

\[
q(\lambda)=\sum_i\min_{x_i\in X_i}
[f_i(x_i)+\lambda^Tg_i(x_i)]-\lambda^TK.
\]

The dual is **maximize** q over lambda >= 0. Each inner minimization is an LP.
Household i knows only its own load, battery data and the price vector; its
linear charging coefficient rises by lambda_t and discharging coefficient
falls by lambda_t. The fixed term lambda^T l_i must be included when computing
the dual value even though it does not affect the household's argmin.

An upper supergradient of the concave q is

\[
s^k=\sum_i g_i(x_i^k)-K.
\]

The customary name is projected dual subgradient ascent:

\[
\lambda^{k+1}=[\lambda^k+\alpha_k s^k]_+.
\]

Overloaded intervals get more expensive; unused intervals get cheaper. Units
are consistent: lambda is EUR/kWh and s is kWh per interval. The step-size
parameter therefore has units corresponding to EUR/kWh², not a power price.

**Strong duality:** the hard-capacity LP is feasible on the supplied scenarios
(idle batteries already fit the transformer) and has a finite optimum, since
local battery variables are bounded. Finite-dimensional LP strong duality
therefore gives max q = P*. Strict Slater feasibility is not required for this
LP theorem. This statement does **not** cover the binary MILP.

For the old soft model, minimizing gamma o - lambda o over o >= 0 would
instead require 0 <= lambda <= gamma. Using only projection onto lambda >= 0
for that soft model would be a different, incorrect dual. We avoid mixing them.

### 9.1 Why we average and recover primal schedules

Local LP optima need not be unique. Near an equilibrium price, a tiny price
change can switch an entire charging block. Consequently, the last primal
iterate can oscillate and violate capacity while dual values are close to
optimal. The implementation records those raw violations rather than hiding them.

We use weighted averages

\[
\bar x^k=\frac{\sum_{j=1}^k\alpha_jx^j}{\sum_{j=1}^k\alpha_j}.
\]

Convexity preserves all local constraints under averaging. A second average
uses only the latter half of the requested iteration budget, reducing initial
transients. Standard diminishing-step convergence assumptions hold for the
chosen alpha_k = a/(N k^0.6): sum alpha_k diverges and sum alpha_k² converges.
A constant-step comparator is included; it need not converge to exact optimality.
The optional Polyak rule uses the best **measured feasible** upper bound, not
the central optimum. It is an empirical option, not asserted to have the same
guarantee as the diminishing schedule.

Averaging alone need not give finite-iteration hard feasibility. First remove
SOC-neutral charging loops locally using Stage 0's proof. Then blend with the
idle schedule x^0. Let b=sum_i l_i and y=sum_i g_i(bar x_i). Set

\[
\beta=\min\left(1,\min_{t:y_t>b_t}\frac{K_t-b_t}{y_t-b_t}\right),\qquad
\hat x_i=x_i^0+\beta(\bar x_i-x_i^0).
\]

For idle-feasible instances this is a feasible convex combination. The
coordinator broadcasts just beta; every household applies it locally. No
central optimizer or centralized optimum is used for recovery. The code
explicitly rejects idle-infeasible recovery instead of silently applying it.
This restriction is a limitation of the current implementation, not of duality.

The best recovered feasible cost P_best is a primal upper bound. The best
q(lambda) is a dual lower bound. Their difference is a numerical optimality
certificate. Stop when it is at most N*tol EUR. Reference objective values are
used only for plots and error reporting, never to choose household actions.

## 10. Sharing ADMM (Boyd et al., Section 7.3)

Introduce a vector z_i of allocated grid energy for each household. With
C = {z: sum_i z_i <= K}, solve

\[
\min_{x_i\in X_i,z}\sum_i f_i(x_i)+I_C(z)
\quad\text{subject to}\quad g_i(x_i)-z_i=0.
\]

This is a **two-block** problem: all x_i collectively form one separable
block, and all z_i form the other. It is not an unqualified multi-block ADMM.
The scaled augmented Lagrangian gives

\[
x_i^{k+1}=\arg\min_{x_i\in X_i}
 f_i(x_i)+\frac\rho2\|g_i(x_i)-z_i^k+u_i^k\|_2^2.
\]

This is a local convex QP, solved with a persistent, warm-started OSQP model.
Only its local data, previous allocation and multiplier are needed.

For v_i = g_i(x_i^{k+1}) + u_i^k, the z update is Euclidean projection onto C:

\[
\delta_t=\left[\frac{\sum_i v_{it}-K_t}{N}\right]_+,
\qquad z_i^{k+1}=v_i-\delta.
\]

Derivation: at a violated interval the projection subtracts the same scalar
from every component; requiring the corrected sum to equal K determines that
scalar. Otherwise the projection is the identity. We do not separately impose
z_i >= 0: auxiliary allocations can be negative during iteration; physical
grid imports remain nonnegative because household constraints ensure that.
At convergence z_i = g_i and the physical interpretation is restored.

Finally,

\[
u_i^{k+1}=u_i^k+g_i(x_i^{k+1})-z_i^{k+1}=\delta.
\]

All scaled multipliers are identical after this step. The coordinator needs
only aggregate imports and broadcasts a single vector delta. Households can
reconstruct their own z_i and u_i; no household receives another's battery
model or raw load profile. This is message-based separation, **not a formal
privacy guarantee**: schedules can reveal information about loads.

### 10.1 Correct residuals and stopping

Let G=[I,-I,0], A=blockdiag(G), B=-I. With the loads absorbed into the equality
constant, the primal residual is r_i = g_i - z_i. The dual residual is

\[
s_i=\rho G^T(z_i^{k+1}-z_i^k)
   =\rho(\Delta z_i,-\Delta z_i,0),
\]

up to a sign that does not affect its norm. Reporting only rho||Delta z||
would omit the G^T factor. The implementation reports both full Euclidean
norms and RMS values, dividing by sqrt(NT) and sqrt(3NT), respectively.
The stopping rule uses absolute RMS tolerances. These alone do not certify a
specified objective gap, so comparison to the independent central reference
is reported separately. Final schedules are locally purified and recovered
with the same idle blend before being called feasible.

The objective functions are closed proper convex, the split constraints are
affine and the supplied feasible LP admits a saddle point. Standard two-block
ADMM theory applies with exact subproblem solves. In practice local QPs are
solved to finite tolerance, so numerical feasibility/objective checks are
also required. Unique battery schedules are not assumed. Tests compare
objective values and feasibility rather than demanding identical arrays.

## 11. Proportional and lexicographic max-min fairness

Define daily saving S_i = -f_i(x_i), relative to an idle battery. This includes
the modelling throughput charge and the terminal energy adjustment. Define
q_i = S_i/B_i in EUR per kWh of installed battery. All fair reference models
impose S_i >= 0 (individual rationality); no negative saving is clipped away.

- Efficient reference: maximize sum_i S_i.
- Proportional fairness: maximize sum_i log(q_i), requiring q_i > 0.
- Max-min fairness: lexicographically maximize the ascending sorted vector q.

Proportional fairness is optimized over battery schedules, hence allocates
scarce transformer capacity over time through those schedules. It is not a
static equal-kW quota. Its first-order characterization is
sum_i (q_i-q_i*)/q_i* <= 0 for all feasible positive q.
Since log(S_i/B_i)=log(S_i)-log(B_i), this unweighted proportional objective
has the **same optimizer** as sum log(S_i). Normalizing the reported savings
does not change that fact. Jain's index of q or the smallest q need not improve
under proportional fairness. The experiments report the actual outcome.

Before solving log fairness, a max-min LP checks that strictly positive
savings for all households are achievable. If not, the code raises; it does
not add an undocumented epsilon to the log. Log arguments are divided by a
fixed positive scale for numerical conditioning, which adds only a constant
to the objective.

Max-min is implemented by progressive freezing: maximize a common floor over
the active households, identify those unable to improve on that optimal face,
fix their achieved level (within a documented numerical tolerance), and
repeat over the remaining households. A final efficiency tie-break chooses
a schedule. This is stronger than merely maximizing one scalar minimum and
leaving all other outcomes arbitrary. Freezing tolerance is 2e-7 EUR/kWh;
bottleneck identification uses ten times that tolerance.

Fairness outputs:

\[
J(q)=\frac{(\sum_i q_i)^2}{N\sum_iq_i^2},\quad
q_{\min}=\min_iq_i,\quad
\mathrm{PoF}=\frac{\sum_iS_i^{\rm efficient}-\sum_iS_i^{\rm fair}}
{\sum_iS_i^{\rm efficient}}.
\]

Jain's index is undefined for an all-zero vector. Price of fairness is
undefined if the efficient benefit is nonpositive. The implementation returns
null in these cases and rejects material negative savings, rather than
manufacturing a favorable metric by clipping.

A distributed proportional-fair variant uses exactly the sharing projection
above, replacing the local linear cost with -log(S_i/B_i). Each household
solves its own convex log-plus-quadratic subproblem. Max-min remains a
**centralized lexicographic reference**; no distributed max-min convergence
claim is made.

## 12. Scaling and communication accounting

Use the same processed BDEW H0 profile and lognormal load-noise mechanism as
the original repository. Six homes use the original parameters. For N=100
and 1000, a fixed generator samples annual usage from the original six levels
and battery sizes from {5,7.5,10} kWh. Load noise uses seed 2025; parameter
sampling uses seed 2042. Every run retains 96 quarter-hour intervals.

Capacity scales as 11*N/6 kW. Keeping 11 kW while increasing household count
100-fold would primarily test infeasibility, not algorithmic scaling.
The scaling date is fixed; separate six-household experiments cover three
2025 dates, including negative prices. No runtime is extrapolated.

Reported runtime is **wall-clock serial simulation** including local oracle
construction, not actual parallel-network runtime. Each iteration conceptually
uses two messages per agent: an uplink containing its T-element grid schedule
and a cost scalar, and a downlink containing a T-element price/projection
vector plus scalar recovery/bookkeeping information. The accounting convention
budgets 2T+3 scalar values and 2 messages per agent per iteration. Hence total
network message count is 2*N*k. This does not measure packetization, latency,
secure aggregation or traffic of a deployed system.

An iteration limit is not convergence. Every output records `converged`,
iteration count, measured runtime, objective gap and feasibility separately.
Tests and plots keep those distinctions visible.

## 13. References and reproducibility

- Boyd et al. (2011), *Distributed Optimization and Statistical Learning via
  the Alternating Direction Method of Multipliers*, Sections 2.2, 3 and 7.3:
  https://web.stanford.edu/~boyd/papers/pdf/admm_distr_stats.pdf
- CVXPY DCP rules: https://www.cvxpy.org/tutorial/dcp/
- HiGHS Python API: https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/
- OSQP Python API: https://osqp.org/docs/interfaces/python.html

Run `python scripts/reproduce_distributed.py` for all experiments and figures.
Alternatively run `python scripts/distributed_experiments.py`,
`python scripts/fair_distributed_experiment.py` and `python scripts/plot_distributed.py`. The experiment script stores CSV histories,
summary JSON and software/data provenance. Stage 0 remains reproducible with
`python scripts/audit_stage0.py --include-milp`. Run `python -m pytest -q` for
all available tests. A full reproduction entry point is documented in README.

Limitations: idealized shared-transformer energy balance; no voltage/reactive
power/network losses; synthetic household loads; perfect day-ahead information;
daily reset and terminal valuation; a modelling rather than calibrated wear
fee; finite solver tolerances; an idle-feasible recovery requirement; central
max-min reference; and serial rather than deployed distributed timing.
