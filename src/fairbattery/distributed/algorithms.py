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
    """Each household solves its LP; coordinator sees imports and one cost scalar.

    Weighted ergodic averaging + feasible idle blending; no centralized solve.
    Polyak uses a measured feasible primal upper bound, not the reference optimum.
    """
    if rule not in ('diminishing','constant','polyak') or step<=0: raise ValueError('invalid step rule')
    start=perf_counter();agents=[HouseholdLP(p,i) for i in range(p.n)]
    lam=np.zeros(p.T);avg=p.idle();weight=0.;best=p.idle();ub=float(p.costs(best).sum())
    lb=-np.inf;history=[];converged=False
    tail=p.idle();tail_weight=0.
    # Checks the recovery precondition immediately.
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
        # Local loop removal and one scalar blending message produce a feasible point.
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
    """Boyd 7.3 sharing; local QPs, aggregate-capacity projection, common dual.

    Split g_i(x_i)=z_i. Coordinator broadcasts delta only; each agent can form
    z_i = g_i + u_old - delta and u_new=delta locally. Execution here is serial.
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
        # A=blockdiag(G), B=-I: true dual residual is rho A^T(znew-z).
        dz=znew-z;s=rho*np.hstack([dz,-dz,np.zeros_like(dz)])
        rnorm=float(np.linalg.norm(r));snorm=float(np.linalg.norm(s))
        u=delta;z=znew
        recovered,_=p.recover(x);value=float(p.costs(recovered).sum())
        if value<ub:best=recovered;ub=value
        if k==1 or k%check_every==0 or k==max_iter:
            # Absolute RMS residual tolerances have comparable meaning across N.
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
