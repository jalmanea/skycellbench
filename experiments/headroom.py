"""Reference upper level: a very long local search with full knowledge of the instance.

  python experiments/headroom.py public [calls] [task ...]
The search sees the users even on the blind task, so the number is what a controller with
unlimited tries and full information reaches, not a baseline. Closed-form scorer.
Appends to results/headroom-<set>.jsonl and skips what is already there.
"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench.baselines import local_search

root = Path(__file__).resolve().parent
name = sys.argv[1]
calls = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
tasks = sys.argv[3:] or list(ub.TASKS)
out = root / "results" / f"headroom-{name}.jsonl"


def run(path):
    inst = ub.Instance.load(path)
    u = local_search(inst, np.random.default_rng([inst.seed, 99]), budget=calls)
    return dict(instance=inst.id, task=inst.task, calls=calls,
                fast=ub.score_fast(inst, u)["score"], uavs=np.round(u, 2).tolist())


if __name__ == "__main__":
    done = {json.loads(l)["instance"] for l in out.open()} if out.exists() else set()
    paths = [p for p in sorted((root.parent / "instances" / name).glob("*.json"))
             if p.stem.split("-")[0] in tasks and ub.Instance.load(p).id not in done]
    with Pool(4) as pool, out.open("a") as f:
        for row in pool.imap_unordered(run, paths):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row["instance"], round(row["fast"], 3), flush=True)
