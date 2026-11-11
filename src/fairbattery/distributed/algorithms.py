"""Price decomposition and two-block sharing ADMM for the hard-capacity LP."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from .local import HouseholdLP, HouseholdQP

@dataclass
class DistributedResult:
    x: np.ndarray
    history: list
    iterations: int
    seconds: float
    converged: bool
    messages_per_agent: int
    scalars_per_agent: int


def dual_decomposition(p, *, max_iter=3000, step=0.1, rule='diminishing', tol=1e-4,
                       reference=None, check_every=10):
    """Dual decomposition on the congestion price for the hard-limit LP.

    Each household solves its own LP for the current price and reports its
    import and cost. Feasible schedules come from step-weighted averages that
    are blended with the idle schedule (theory, Section 13). The run stops
    when the best feasible cost and the best dual bound are within
    ``tol * n`` EUR. ``reference`` is only used to log the gap. The Polyak
    rule uses the best feasible cost found so far, not the true optimum.
    """
    if rule not in ('diminishing','constant','polyak') or step<=0: raise ValueError('invalid step rule')
    start=perf_counter();agents=[HouseholdLP(p,i) for i in range(p.n)]
    lam=np.zeros(p.T);avg=p.idle();weight=0.;best=p.idle();ub=float(p.costs(best).sum())
    lb=-np.inf;history=[];converged=False
    tail=p.idle();tail_weight=0.
    # Fail early if the idle schedule is infeasible; recovery depends on it.
    p.recover(best)
    for k in range(1,max_iter+1):
        answers=[a.solve(lam) for a in agents]
        x=np.stack([a[0] for a in answers]);grids=np.stack([a[1] for a in answers])
        cost=sum(a[2] for a in answers);s=grids.sum(axis=0)-p.limit
        dual=float(cost+lam@s);lb=max(lb,dual)
        if rule=='polyak':
            alpha=step*max(ub-dual,1e-12)/max(float(s@s),1e-20)
        elif rule=='diminishing': alpha=step/(k**0.6*p.n)
        else: alpha=step/p.n
        weight+=alpha;avg+=(alpha/weight)*(x-avg)
        # Recover a feasible schedule from the last iterate and from each average.
        candidates=[x,avg]
        if k>max_iter//2:
            tail_weight+=alpha;tail+=(alpha/tail_weight)*(x-tail)
            candidates.append(tail)
        for candidate in candidates:
            recovered,_=p.recover(candidate);value=float(p.costs(recovered).sum())
            if value<ub:ub=value;best=recovered
        next_lam=np.maximum(0,lam+alpha*s)
        if k==1 or k%check_every==0 or k==max_iter:
            gap=max(0.,ub-lb)
            history.append(dict(iteration=k,objective=ub,dual_bound=lb,certificate_gap=gap,
                                objective_gap=ub-reference if reference is not None else None,
                                raw_objective=float(p.costs(x).sum()),
                                raw_violation_kw=float(np.maximum(s,0).max()/p.dt),
                                average_violation_kw=p.diagnostics(avg)['max_capacity_violation_kw'],
                                violation_kw=p.diagnostics(best)['max_capacity_violation_kw'],
                                price_change=float(np.linalg.norm(next_lam-lam)),step=alpha))
            if gap<=tol*p.n:
                converged=True;break
        lam=next_lam
    return DistributedResult(best,history,k,perf_counter()-start,converged,
                             2*k,k*(2*p.T+3))


def sharing_admm(p, *, rho=0.1, max_iter=2000, tol=1e-4, reference=None, check_every=10):
    """Two-block sharing ADMM (Boyd et al. 2011, Section 7.3).

    Households solve local QPs with the split g_i(x_i) = z_i, and the
    coordinator projects the allocations onto the capacity limit. Every
    household ends up with the same scaled multiplier ``delta``, so the
    coordinator only broadcasts that vector; each household forms
    z_i = g_i + u_old - delta itself. Runs serially here.
    """
    if rho<=0:raise ValueError('rho must be positive')
    start=perf_counter();agents=[HouseholdQP(p,i,rho) for i in range(p.n)]
    z=p.loads.copy();u=np.zeros(p.T);best=p.idle();ub=float(p.costs(best).sum())
    history=[];converged=False;p.recover(best)
    for k in range(1,max_iter+1):
        answers=[a.solve(z[i],u) for i,a in enumerate(agents)]
        x=np.stack([a[0] for a in answers]);g=np.stack([a[1] for a in answers])
        v=g+u
        delta=np.maximum((v.sum(axis=0)-p.limit)/p.n,0)
        znew=v-delta
        r=g-znew
        # With A = blockdiag(G) and B = -I, the dual residual is rho * A^T (z_new - z).
        dz=znew-z;s=rho*np.hstack([dz,-dz,np.zeros_like(dz)])
        rnorm=float(np.linalg.norm(r));snorm=float(np.linalg.norm(s))
        u=delta;z=znew
        recovered,_=p.recover(x);value=float(p.costs(recovered).sum())
        if value<ub:best=recovered;ub=value
        if k==1 or k%check_every==0 or k==max_iter:
            # RMS residuals, so one tolerance works for any number of households.
            rr=rnorm/np.sqrt(p.n*p.T);ss=snorm/np.sqrt(3*p.n*p.T)
            history.append(dict(iteration=k,objective=ub,
                                objective_gap=ub-reference if reference is not None else None,
                                primal_residual=rnorm,dual_residual=snorm,
                                primal_rms=rr,dual_rms=ss,
                                raw_objective=float(p.costs(x).sum()),
                                raw_violation_kw=p.diagnostics(x)['max_capacity_violation_kw'],
                                violation_kw=p.diagnostics(best)['max_capacity_violation_kw'],rho=rho))
            if rr<=tol and ss<=tol:
                converged=True;break
    return DistributedResult(best,history,k,perf_counter()-start,converged,
                             2*k,k*(2*p.T+3))
