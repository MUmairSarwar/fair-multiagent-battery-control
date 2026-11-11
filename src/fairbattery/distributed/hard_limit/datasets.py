"""Fixed-seed BDEW H0 scenarios using the repository's existing load method."""
import numpy as np
from fairbattery import data
from fairbattery.env import BatteryCommunity,EnvConfig
from .model import DayProblem


def scenario(n=6,date='2025-01-07',seed=2025):
    dates,prices,h0=data.load_days()
    match=np.flatnonzero(dates==date)
    if not len(match):raise ValueError('Date absent or excluded DST day: '+date)
    if n==6:
        cfg=EnvConfig()
    else:
        rng=np.random.default_rng(seed+17)
        cfg=EnvConfig(annual_kwh=rng.choice([2500,3000,3500,4000,4500,5000],n).tolist(),
                      capacity_kwh=rng.choice([5.,7.5,10.],n).tolist(),
                      transformer_kw=11*n/6)
    env=BatteryCommunity(cfg,prices,h0)
    env.reset(match,np.random.default_rng(seed))
    return DayProblem.from_env(env)
