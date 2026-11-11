import numpy as np
import pytest
from fairbattery.distributed.hard_limit.datasets import scenario
from fairbattery.distributed.hard_limit.model import DayProblem,central_lp,central_cvxpy
from fairbattery.distributed.hard_limit.algorithms import dual_decomposition,sharing_admm
from fairbattery.distributed.hard_limit.fairness import solve_fair,metrics

@pytest.mark.parametrize('date',['2025-01-07','2025-05-14','2025-08-06'])
def test_independent_central_formulations_agree(date):
    p=scenario(date=date);x,v=central_lp(p);y,w=central_cvxpy(p)
    assert abs(v-w)<1e-6
    assert p.diagnostics(x)['max_capacity_violation_kw']<1e-7
    assert p.diagnostics(y)['local_violation_kwh']<1e-6

@pytest.mark.parametrize('date',['2025-01-07','2025-05-14','2025-08-06'])
def test_admm_matches_cvxpy(date):
    p=scenario(date=date);_,v=central_cvxpy(p)
    r=sharing_admm(p,rho=.1,max_iter=2000,tol=1e-5,reference=v)
    diag=p.diagnostics(r.x)
    assert abs(diag['objective']-v)<.003
    assert diag['max_capacity_violation_kw']<1e-6
    assert diag['local_violation_kwh']<5e-5
    assert diag['overlap_steps']==0
    assert r.converged
    assert r.messages_per_agent==2*r.iterations


def test_dual_averaging_and_recovery_match_cvxpy():
    p=scenario();_,v=central_cvxpy(p)
    r=dual_decomposition(p,max_iter=3000,step=.03,tol=3e-4,reference=v)
    assert abs(p.costs(r.x).sum()-v)<.003
    assert p.diagnostics(r.x)['max_capacity_violation_kw']<1e-7
    assert p.diagnostics(r.x)['local_violation_kwh']<1e-7
    assert r.history[-1]['dual_bound']<=v+1e-6
    assert r.history[-1]['raw_violation_kw']>r.history[-1]['violation_kw']


def test_recovery_preserves_soc_and_restores_capacity():
    p=scenario();x=p.idle();T=p.T
    # Household LP solutions that are each feasible but jointly exceed the limit.
    from fairbattery.distributed.hard_limit.local import HouseholdLP
    x=np.stack([HouseholdLP(p,i).solve(np.zeros(T))[0] for i in range(p.n)])
    y,a=p.recover(x)
    assert 0<=a<=1
    assert p.diagnostics(y)['max_capacity_violation_kw']<1e-7
    assert p.diagnostics(y)['local_violation_kwh']<1e-7


def test_fairness_and_price_of_fairness():
    p=scenario(date='2025-05-14')
    xe,me=solve_fair(p,'efficient');xp,mp=solve_fair(p,'proportional');xm,mm=solve_fair(p,'max_min')
    assert np.log(mp['saving_per_kwh']).sum()>=np.log(me['saving_per_kwh']).sum()-1e-6
    assert mm['min_saving_per_kwh']>=me['min_saving_per_kwh']-1e-6
    for x in (xp,xm):
        m=metrics(p,x,me['total_saving_eur'])
        assert m['price_of_fairness']>=-1e-6
        assert 1/p.n-1e-6<=m['jain']<=1+1e-6
        assert m['max_capacity_violation_kw']<1e-6


def test_log_fairness_rejects_zero_surplus():
    p=DayProblem(np.ones(4),np.ones((2,4))*.1,np.ones(2),np.ones(2),np.ones(4)*2)
    with pytest.raises(ValueError,match='positive savings'):
        solve_fair(p,'proportional')


def test_scaled_h0_scenarios_are_reproducible_and_capacity_scales():
    a=scenario(n=100);b=scenario(n=100)
    np.testing.assert_array_equal(a.loads,b.loads)
    assert a.n==100 and a.T==96
    np.testing.assert_allclose(a.limit,11*100/6*.25)
    assert np.all(a.loads.sum(axis=0)<a.limit)


def test_idle_infeasible_recovery_fails_explicitly():
    p=scenario();p.limit[:]=.001
    with pytest.raises(ValueError,match='Idle-feasible'):
        p.recover(p.idle())


def test_distributed_log_savings_matches_central_reference():
    from fairbattery.distributed.hard_limit.fairness import distributed_proportional
    p=scenario(date='2025-05-14');x,_=solve_fair(p,'proportional')
    r=distributed_proportional(p,rho=.1,max_iter=2000,tol=1e-6)
    central=np.log(-p.costs(x)/p.capacities).sum()
    obtained=np.log(-p.costs(r.x)/p.capacities).sum()
    assert r.converged
    assert abs(central-obtained)<1e-3
    assert p.diagnostics(r.x)['max_capacity_violation_kw']<1e-6
    assert p.diagnostics(r.x)['local_violation_kwh']<1e-5
