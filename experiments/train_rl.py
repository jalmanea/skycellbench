"""Train the PPO baseline for one task on the closed-form scorer.

  python experiments/train_rl.py place 300000
Instances are drawn fresh from the generator (seeds below 90000, never the public or held-out
sets). Writes models/rl-<task>.pt and a learning curve to results/rl-<task>.jsonl.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import skycellbench as ub
from skycellbench.rl import STEPS, PlacementEnv, Policy

task, total = sys.argv[1], int(sys.argv[2])
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
root = Path(__file__).resolve().parent
(root / "models").mkdir(exist_ok=True)
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)

N_EP, GAMMA, LAM, CLIP, EPOCHS, BATCH, LR, R_SCALE = 32, 0.99, 0.95, 0.2, 6, 512, 3e-4, 10.0
policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=LR)


def new_env():
    k, n = rng.choice([3, 5, 8]), rng.choice([150, 200, 250])
    inst = ub.generate(int(rng.integers(90000)), task, n_uav=int(k), n_users=int(n))
    probe = None
    if task == "blind":
        probe = ub.Probe(inst)
    return PlacementEnv(inst.public(), probe), inst


log = (root / "results" / f"rl-{task}.jsonl").open("w")
steps, t0, finals = 0, time.time(), []
while steps < total:
    buf = {k: [] for k in ("obs", "act", "logp", "ret", "adv")}
    for _ in range(N_EP):
        env, inst = new_env()
        obs, done = env.reset(), False
        ep = {k: [] for k in ("obs", "act", "logp", "val", "rew")}
        while not done:
            o = torch.as_tensor(obs)
            with torch.no_grad():
                d = policy.dist(o)
                a = d.sample()
                ep["logp"].append(d.log_prob(a).sum(-1).numpy())
                ep["val"].append(policy.v(o).squeeze(-1).numpy())
            ep["obs"].append(obs)
            ep["act"].append(a.numpy())
            obs, r, done = env.step(np.tanh(a.numpy()))
            ep["rew"].append(r * R_SCALE)
        finals.append(ub.score_fast(inst, env.answer())["score"])
        val = np.vstack(ep["val"] + [np.zeros(env.k)])          # (STEPS + 1, K)
        adv, last = np.zeros((STEPS, env.k)), np.zeros(env.k)
        for t in reversed(range(STEPS)):
            delta = ep["rew"][t] + GAMMA * val[t + 1] - val[t]
            adv[t] = last = delta + GAMMA * LAM * last
        buf["obs"].append(np.concatenate(ep["obs"]))
        buf["act"].append(np.concatenate(ep["act"]))
        buf["logp"].append(np.concatenate(ep["logp"]))
        buf["adv"].append(adv.ravel())
        buf["ret"].append((adv + val[:-1]).ravel())
        steps += STEPS
    b = {k: torch.as_tensor(np.concatenate(v), dtype=torch.float32) for k, v in buf.items()}
    b["adv"] = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
    for _ in range(EPOCHS):
        for idx in torch.randperm(len(b["obs"])).split(BATCH):
            d = policy.dist(b["obs"][idx])
            ratio = (d.log_prob(b["act"][idx]).sum(-1) - b["logp"][idx]).exp()
            pg = -torch.min(ratio * b["adv"][idx],
                            ratio.clamp(1 - CLIP, 1 + CLIP) * b["adv"][idx]).mean()
            vl = (policy.v(b["obs"][idx]).squeeze(-1) - b["ret"][idx]).pow(2).mean()
            opt.zero_grad()
            (pg + 0.5 * vl).backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
            opt.step()
    if len(finals) >= 160:
        row = dict(steps=steps, score=float(np.mean(finals)), minutes=(time.time() - t0) / 60)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        finals = []
        torch.save(policy.state_dict(), root / "models" / f"rl-{task}.pt")
torch.save(policy.state_dict(), root / "models" / f"rl-{task}.pt")
