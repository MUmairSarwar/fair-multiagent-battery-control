"""Learning agents: independent tabular Q-learning, independent PPO and MAPPO.

All agents act only on their own observation. The only difference between
IPPO and MAPPO is the critic used during training: in IPPO each agent has its
own critic that sees its own observation, in MAPPO one shared critic sees the
observations of all agents ("centralised training, decentralised execution").
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .data import STEPS_PER_DAY
from .env import BatteryCommunity

# ======================================================================
# Tabular Q-learning
# ======================================================================


class TabularQ:
    """One Q-table per agent. The observation is cut into a few bins:
    battery level (5) x time of day (6 blocks of 4 h) x share of the rest of
    the day that is cheaper than now (5) x transformer busy last step (2) x
    other agents signalled last step (2)."""

    name = "iql"
    SOC_BINS, HOUR_BINS, CHEAP_BINS = 5, 6, 5

    def __init__(self, env: BatteryCommunity, rng, alpha=0.1, gamma=0.99):
        self.n, self.nA = env.n, env.n_actions
        self.n_states = self.SOC_BINS * self.HOUR_BINS * self.CHEAP_BINS * 2 * 2
        self.q = np.zeros((self.n, self.n_states, self.nA))
        self.alpha, self.gamma = alpha, gamma
        self.rng = rng
        self.eps = 0.0

    def state(self, env: BatteryCommunity, obs: np.ndarray) -> np.ndarray:
        soc = np.minimum((obs[..., 0] * self.SOC_BINS).astype(int), self.SOC_BINS - 1)
        t = min(env.t, STEPS_PER_DAY - 1)
        hour = np.full(soc.shape, t * self.HOUR_BINS // STEPS_PER_DAY)
        cheap = np.minimum((obs[..., 5] * self.CHEAP_BINS).astype(int), self.CHEAP_BINS - 1)
        busy = (obs[..., 9] > 0.9).astype(int)
        talk = (obs[..., 10] > 0).astype(int)
        s = (((soc * self.HOUR_BINS + hour) * self.CHEAP_BINS + cheap) * 2 + busy) * 2 + talk
        return s

    def act(self, env, obs, greedy=True):
        s = self.state(env, obs)
        q = self.q[np.arange(self.n)[None, :], s]  # [B, n, A]
        # random tie-breaking so that an untrained table does not always pick action 0
        noise = self.rng.random(q.shape) * 1e-9
        a = (q + noise).argmax(axis=-1)
        if not greedy and self.eps > 0:
            explore = self.rng.random(a.shape) < self.eps
            a = np.where(explore, self.rng.integers(0, self.nA, a.shape), a)
        return a

    def train(self, env: BatteryCommunity, train_days, n_iters, batch_days, log=None):
        agent_idx = np.arange(self.n)[None, :]
        for it in range(n_iters):
            self.eps = max(0.05, 1.0 - it / (0.8 * n_iters))
            days = self.rng.choice(train_days, size=batch_days)
            obs = env.reset(days, self.rng)
            s = self.state(env, obs)
            done = False
            ep_ret = 0.0
            while not done:
                a = self.act(env, obs, greedy=False)
                obs, r, done, _ = env.step(a)
                ep_ret += r.sum(axis=1).mean()
                s2 = self.state(env, obs)
                target = r + (0.0 if done else self.gamma) * self.q[agent_idx, s2].max(axis=-1)
                # several days in the batch can hit the same (agent, state, action);
                # their TD errors are averaged so that one entry moves by at most alpha
                td = target - self.q[agent_idx, s, a]
                key = (np.broadcast_to(agent_idx, s.shape) * self.n_states + s) * self.nA + a
                td_sum = np.bincount(key.ravel(), weights=td.ravel(), minlength=self.q.size)
                count = np.bincount(key.ravel(), minlength=self.q.size)
                hit = count > 0
                flat = self.q.reshape(-1)
                flat[hit] += self.alpha * td_sum[hit] / count[hit]
                s = s2
            if log is not None:
                log.append({"iter": it, "return": float(ep_ret)})


# ======================================================================
# PPO (independent and centralised critic)
# ======================================================================


def mlp(inp, out, hidden=64):
    return nn.Sequential(nn.Linear(inp, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))


class PPOAgents(nn.Module):
    def __init__(self, env: BatteryCommunity, centralised_critic: bool, seed: int, lr=3e-4,
                 gamma=0.99, lam=0.95, clip=0.2, epochs=4, minibatches=4, ent_coef=0.01,
                 vf_coef=0.5, hidden=64):
        super().__init__()
        torch.manual_seed(seed)
        self.n, self.nA, self.obs_dim = env.n, env.n_actions, env.obs_dim
        self.centralised = centralised_critic
        self.name = "mappo" if centralised_critic else "ippo"
        self.actors = nn.ModuleList([mlp(self.obs_dim, self.nA, hidden) for _ in range(self.n)])
        if centralised_critic:
            self.critic = mlp(self.obs_dim * self.n, self.n, 2 * hidden)
        else:
            self.critics = nn.ModuleList([mlp(self.obs_dim, 1, hidden) for _ in range(self.n)])
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)
        self.gamma, self.lam, self.clip = gamma, lam, clip
        self.epochs, self.minibatches = epochs, minibatches
        self.ent_coef, self.vf_coef = ent_coef, vf_coef

    # obs: tensor [..., n, obs_dim]
    def logits(self, obs):
        return torch.stack([self.actors[i](obs[..., i, :]) for i in range(self.n)], dim=-2)

    def values(self, obs):
        if self.centralised:
            flat = obs.reshape(*obs.shape[:-2], self.n * self.obs_dim)
            return self.critic(flat)
        return torch.cat([self.critics[i](obs[..., i, :]) for i in range(self.n)], dim=-1)

    @torch.no_grad()
    def act(self, env, obs, greedy=True):
        lg = self.logits(torch.as_tensor(obs, dtype=torch.float32))
        if greedy:
            return lg.argmax(-1).numpy()
        return torch.distributions.Categorical(logits=lg).sample().numpy()

    def train_agents(self, env: BatteryCommunity, train_days, n_iters, batch_days, rng, log=None):
        T = STEPS_PER_DAY
        for it in range(n_iters):
            days = rng.choice(train_days, size=batch_days)
            obs = env.reset(days, rng)
            buf_obs, buf_act, buf_logp, buf_rew, buf_val = [], [], [], [], []
            done = False
            while not done:
                o = torch.as_tensor(obs, dtype=torch.float32)
                with torch.no_grad():
                    lg = self.logits(o)
                    dist = torch.distributions.Categorical(logits=lg)
                    a = dist.sample()
                    v = self.values(o)
                buf_obs.append(o)
                buf_act.append(a)
                buf_logp.append(dist.log_prob(a))
                buf_val.append(v)
                obs, r, done, _ = env.step(a.numpy())
                buf_rew.append(torch.as_tensor(r, dtype=torch.float32))
            O = torch.stack(buf_obs)  # [T, B, n, d]
            A = torch.stack(buf_act)  # [T, B, n]
            LP = torch.stack(buf_logp)
            R = torch.stack(buf_rew)
            V = torch.stack(buf_val)
            # GAE; the episode ends after T steps, so the value after the last step is 0
            adv = torch.zeros_like(R)
            last = torch.zeros_like(R[0])
            for t in reversed(range(T)):
                nv = V[t + 1] if t + 1 < T else torch.zeros_like(V[0])
                delta = R[t] + self.gamma * nv - V[t]
                last = delta + self.gamma * self.lam * last
                adv[t] = last
            ret = adv + V
            # flatten time and batch
            O, A, LP, adv, ret = (x.reshape(T * batch_days, *x.shape[2:]) for x in (O, A, LP, adv, ret))
            adv_n = (adv - adv.mean(0)) / (adv.std(0) + 1e-8)
            N = O.shape[0]
            mb = N // self.minibatches
            for _ in range(self.epochs):
                perm = torch.randperm(N)
                for k in range(self.minibatches):
                    idx = perm[k * mb : (k + 1) * mb]
                    lg = self.logits(O[idx])
                    dist = torch.distributions.Categorical(logits=lg)
                    lp = dist.log_prob(A[idx])
                    ratio = torch.exp(lp - LP[idx])
                    s1 = ratio * adv_n[idx]
                    s2 = torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * adv_n[idx]
                    pol_loss = -torch.min(s1, s2).mean()
                    v_loss = ((self.values(O[idx]) - ret[idx]) ** 2).mean()
                    ent = dist.entropy().mean()
                    loss = pol_loss + self.vf_coef * v_loss - self.ent_coef * ent
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.parameters(), 0.5)
                    self.opt.step()
            if log is not None:
                log.append({"iter": it, "return": float(R.sum(0).sum(-1).mean())})
