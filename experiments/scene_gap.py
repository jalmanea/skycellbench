"""Does a real scene make position matter more? k-means against a long reference search.

  python experiments/scene_gap.py <set> [calls] [task ...]     e.g. public-cincinnati-downtown
For every instance of the set, on the scene and again on the old statistical channel with the
same users (the instance with its scene removed, so flat ground and no buildings): the score of
the k-means answer (best of 3 altitudes) and of a local search with `calls` scorer calls.
Closed-form scorer. Appends to results/scene-gap-<set>.jsonl and skips what is already there.
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench import controllers
from skycellbench.baselines import local_search

root = Path(__file__).resolve().parent
name = sys.argv[1]
calls = int(sys.argv[2]) if len(sys.argv) > 2 else 200_000
tasks = sys.argv[3:] or ["place", "constrained"]
out = root / "results" / f"scene-gap-{name}.jsonl"


def run(job):
    path, channel = job
    inst = ub.Instance.load(path)
    if channel == "statistical":
        inst = dataclasses.replace(inst, scene=None)
    km = controllers.kmeans(inst, None, np.random.default_rng([inst.seed, 6]))
    ref = local_search(inst, np.random.default_rng([inst.seed, 99]), budget=calls)
    return dict(instance=inst.id, task=inst.task, channel=channel, calls=calls,
                kmeans=ub.score_fast(inst, km)["score"], reference=ub.score_fast(inst, ref)["score"],
                kmeans_uavs=np.round(km, 2).tolist(), reference_uavs=np.round(ref, 2).tolist())


if __name__ == "__main__":
    done = set()
    if out.exists():
        done = {(r["instance"], r["channel"], r["calls"]) for r in map(json.loads, out.open())}
    jobs = [(p, c) for p in sorted((root.parent / "instances" / name).glob("*.json"))
            if p.stem.split("-")[0] in tasks for c in ("scene", "statistical")
            if (ub.Instance.load(p).id, c, calls) not in done]
    with Pool(3) as pool, out.open("a") as f:
        for row in pool.imap_unordered(run, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row["instance"], row["channel"], round(row["kmeans"], 3),
                  round(row["reference"], 3), flush=True)
