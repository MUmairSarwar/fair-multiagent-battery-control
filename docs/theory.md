# Distributed optimization: Stage 0 study notes

**Authorship status:** AI-assisted learning draft, not yet an independently
written author derivation. Muhammad Umair Sarwar should verify each argument
and rewrite it in his own words before describing the derivations as authored
by him. The implementation was built with AI assistance.

Only Stage 0 is implemented. Stages 1–4 will be developed after the author
reviews and understands this stage.

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
versus hard capacity before deriving Stage 1.

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
slack also changes the dual price domain. None of that is implemented yet.

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
