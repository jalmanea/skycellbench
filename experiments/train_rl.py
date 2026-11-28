"""Train the PPO baseline for one task on the closed-form scorer.

  python experiments/train_rl.py place 300000 [seed] [--scene cincinnati-downtown]
Instances are drawn fresh from the generator (seeds below 90000, never the public or held-out
sets), on flat ground or on the scene. Writes models/rl-<task>[-<scene>].pt and a learning curve
to results/rl-<task>[-<scene>].jsonl.

Every time a point of the curve is written, the policy, the optimiser and both random states are
saved to models/rl-<task>[-<scene>].ckpt. If that file exists the run continues from it, so a run
that is stopped picks up where it was and ends as an uninterrupted one would. Delete it to start
again.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

import skycellbench as ub
from skycellbench.rl import STEPS, PlacementEnv, Policy

args = sys.argv[1:]
scene = None
if "--scene" in args:
    i = args.index("--scene")
    scene = args[i + 1]
    del args[i:i + 2]
task, total = args[0], int(args[1])
seed = int(args[2]) if len(args) > 2 else 0
name = f"rl-{task}" + (f"-{scene}" if scene else "")
root = Path(__file__).resolve().parent
(root / "models").mkdir(exist_ok=True)
MODEL, CKPT = root / "models" / f"{name}.pt", root / "models" / f"{name}.ckpt"
rng = np.random.default_rng(seed)
torch.manual_seed(seed)
torch.set_num_threads(1)

N_EP, GAMMA, LAM, CLIP, EPOCHS, BATCH, LR, R_SCALE = 32, 0.99, 0.95, 0.2, 6, 512, 3e-4, 10.0
policy = Policy()
opt = torch.optim.Adam(policy.parameters(), lr=LR)


def new_env():
    k, n = rng.choice([3, 5, 8]), rng.choice([150, 200, 250])
    inst = ub.generate(int(rng.integers(90000)), task, n_uav=int(k), n_users=int(n), scene=scene)
    probe = None
    if task == "blind":
        probe = ub.Probe(inst)
    return PlacementEnv(inst.public(), probe), inst


steps, before, finals = 0, 0.0, []                 # before: minutes spent in earlier runs
if CKPT.exists():
    c = torch.load(CKPT, weights_only=False)
    assert (c["task"], c["scene"], c["seed"]) == (task, scene, seed), "checkpoint of another run"
    policy.load_state_dict(c["policy"])
    opt.load_state_dict(c["opt"])
    rng.bit_generator.state = c["rng"]
    torch.set_rng_state(c["torch_rng"])
    steps, before = c["steps"], c["minutes"]
    print(f"continuing {name} from {steps} steps", flush=True)
log = (root / "results" / f"{name}.jsonl").open("a" if steps else "w")
t0 = time.time()
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
        minutes = before + (time.time() - t0) / 60
        row = dict(steps=steps, score=float(np.mean(finals)), minutes=minutes)
        log.write(json.dumps(row) + "\n")
        log.flush()
        print(json.dumps(row), flush=True)
        finals = []
        torch.save(policy.state_dict(), MODEL)
        torch.save(dict(task=task, scene=scene, seed=seed, steps=steps, minutes=minutes,
                        policy=policy.state_dict(), opt=opt.state_dict(),
                        rng=rng.bit_generator.state, torch_rng=torch.get_rng_state()), CKPT)
torch.save(policy.state_dict(), MODEL)
