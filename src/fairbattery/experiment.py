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
        out[pol.name], _ = evaluate_policy(env, pol, test_days, loads)
    env.reset(test_days, np.random.default_rng(0), loads=loads)
    bc = baseline_cost(env)
    lp_res = run_lp(env)
    out["lp"] = summarise(lp_res, env, bc)
    out["lp"]["simultaneous_charge_discharge_steps"] = lp_res["simultaneous_charge_discharge_steps"]
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
    metrics, _ = evaluate_policy(env, GreedyWrapper(agent), test_days, loads)
    metrics.update({"algo": algo, "seed": seed, "train_seconds": round(train_seconds, 1)})
    return {"metrics": metrics, "log": log}


def run_all(cfg: dict, exp: dict, out_dir: Path = RESULTS_DIR, algos=None, variants=None, seeds=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    algos = algos or exp["algorithms"]
    variants = variants or list(exp["variants"])
    seeds = seeds if seeds is not None else cfg["train"]["seeds"]
    baselines = {}
    for v in variants:
        baselines[v] = run_baselines(cfg, exp["variants"][v])
    (out_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))
    for v in variants:
        for algo in algos:
            for seed in seeds:
                path = runs_dir / f"{v}__{algo}__seed{seed}.json"
                res = train_and_eval(copy.deepcopy(cfg), algo, seed, exp["variants"][v])
                res["metrics"]["variant"] = v
                path.write_text(json.dumps(res))
                m = res["metrics"]
                print(f"{v:14s} {algo:6s} seed {seed}: net benefit {m['net_benefit_eur']:8.2f} EUR, "
                      f"overload steps {m['overload_steps']:4d}, jain {m['jain']:.3f} "
                      f"({m['train_seconds']:.0f}s)", flush=True)
