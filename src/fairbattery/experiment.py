"""Train and evaluate one (algorithm, variant, seed) combination."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from . import data
from .agents import PPOAgents, TabularQ
from .baselines import NoBatteryPolicy, PriceRulePolicy, run_lp
from .env import BatteryCommunity, EnvConfig
from .metrics import baseline_cost, rollout, summarise

CONFIG_DIR = data.REPO_ROOT / "configs"
RESULTS_DIR = data.REPO_ROOT / "results"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load((path or CONFIG_DIR / "default.yaml").read_text())


def make_env(cfg: dict, overrides: dict | None = None):
    env_kwargs = dict(cfg["env"])
    env_kwargs.update(overrides or {})
    dates, price, h0 = data.load_days()
    env = BatteryCommunity(EnvConfig(**env_kwargs), price, h0)
    train_days, test_days = data.split_days(len(dates))
    return env, train_days, test_days


def test_loads(env: BatteryCommunity, test_days, seed: int) -> np.ndarray:
    env.reset(test_days, np.random.default_rng(seed))
    return env.loads.copy()


class GreedyWrapper:
    def __init__(self, agent):
        self.agent = agent
        self.name = agent.name

    def act(self, env, obs):
        return self.agent.act(env, obs, greedy=True)


def evaluate_policy(env, policy, test_days, loads):
    res = rollout(env, policy, test_days, np.random.default_rng(0), loads=loads)
    bc = baseline_cost(env)
    return summarise(res, env, bc), res


def run_baselines(cfg: dict, overrides: dict | None = None) -> dict:
    env, _, test_days = make_env(cfg, overrides)
    loads = test_loads(env, test_days, cfg["eval"]["load_seed"])
    out = {}
    for pol in [NoBatteryPolicy(), PriceRulePolicy()]:
        m, res = evaluate_policy(env, pol, test_days, loads)
        out[pol.name] = {"metrics": m, "detail": detail(res)}
    env.reset(test_days, np.random.default_rng(0), loads=loads)
    bc = baseline_cost(env)
    lp_res = run_lp(env)
    m = summarise(lp_res, env, bc)
    m["simultaneous_charge_discharge_steps"] = lp_res["simultaneous_charge_discharge_steps"]
    out["lp"] = {"metrics": m, "detail": detail(lp_res)}
    out["price_eur_mwh"] = np.round(env.price * 1000, 2).tolist()
    out["base_load_kw"] = np.round(loads.sum(axis=1) / 0.25, 3).tolist()
    out["test_days"] = [str(d) for d in data.load_days()[0][test_days]]
    return out


def train_and_eval(cfg: dict, algo: str, seed: int, overrides: dict | None = None) -> dict:
    env, train_days, test_days = make_env(cfg, overrides)
    loads = test_loads(env, test_days, cfg["eval"]["load_seed"])
    tcfg = cfg["train"]
    rng = np.random.default_rng(1000 + seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)  # small networks; one thread is faster and deterministic
    log: list = []
    t0 = time.time()
    if algo == "iql":
        p = tcfg["iql"]
        agent = TabularQ(env, rng, alpha=p["alpha"], gamma=p["gamma"])
        agent.train(env, train_days, p["iterations"], tcfg["batch_days"], log=log)
    elif algo in ("ippo", "mappo"):
        p = tcfg["ppo"]
        agent = PPOAgents(
            env, centralised_critic=(algo == "mappo"), seed=seed, lr=p["lr"], gamma=p["gamma"],
            lam=p["lam"], clip=p["clip"], epochs=p["epochs"], minibatches=p["minibatches"],
            ent_coef=p["ent_coef"], hidden=p["hidden"],
        )
        agent.train_agents(env, train_days, p["iterations"], tcfg["batch_days"], rng, log=log)
    else:
        raise ValueError(algo)
    train_seconds = time.time() - t0
    metrics, res = evaluate_policy(env, GreedyWrapper(agent), test_days, loads)
    metrics.update({"algo": algo, "seed": seed, "train_seconds": round(train_seconds, 1)})
    return {"metrics": metrics, "log": log, "detail": detail(res)}


def detail(res: dict) -> dict:
    """Small per-day/per-household arrays kept for the plots."""
    return {
        "total_kw": np.round(res["total_kw"], 3).tolist(),  # [days, 96]
        "saving_house_eur": np.round(res["saving"].sum(axis=(0, 2)), 4).tolist(),
    }


def _job(args):
    cfg, v, overrides, algo, seed, path = args
    res = train_and_eval(cfg, algo, seed, overrides)
    res["metrics"]["variant"] = v
    Path(path).write_text(json.dumps(res))
    m = res["metrics"]
    return (f"{v:14s} {algo:6s} seed {seed}: net benefit {m['net_benefit_eur']:8.2f} EUR, "
            f"overload steps {m['overload_steps']:4d}, jain {m['jain']:.3f} ({m['train_seconds']:.0f}s)")


def run_all(cfg: dict, exp: dict, out_dir: Path = RESULTS_DIR, workers: int = 1):
    """Train every (variant, algorithm, seed) and write one JSON file per run."""
    from multiprocessing import get_context

    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    baselines = {v: run_baselines(cfg, o) for v, o in exp["variants"].items()}
    (out_dir / "baselines.json").write_text(json.dumps(baselines))
    print("baselines done", flush=True)
    jobs = [
        (copy.deepcopy(cfg), v, o, algo, seed, str(runs_dir / f"{v}__{algo}__seed{seed}.json"))
        for v, o in exp["variants"].items()
        for algo in exp["algorithms"]
        for seed in cfg["train"]["seeds"]
    ]
    if workers <= 1:
        for j in jobs:
            print(_job(j), flush=True)
    else:
        with get_context("spawn").Pool(workers) as pool:
            for line in pool.imap_unordered(_job, jobs):
                print(line, flush=True)
