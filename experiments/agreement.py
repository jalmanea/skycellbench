"""Do the two scorers rank placements the same way?

For a few instances, score the same placements with the closed-form scorer and with ns-3:
random placements, k-means at several altitudes, perturbed k-means, and the placements a local
search on the closed-form scorer passes through. Appends one JSON line per placement to
results/agreement.jsonl; rerunning skips what is already there.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench.baselines import clip, kmeans_placement, local_search, random_placement

OUT = Path(__file__).parent / "results" / "agreement.jsonl"
CONFIGS = [dict(seed=101, n_uav=5, n_users=200), dict(seed=102, n_uav=3, n_users=150),
           dict(seed=103, n_uav=8, n_users=250)]


def placements(inst, rng):
    out = [("random", random_placement(inst, rng)) for _ in range(14)]
    for alt in (50, 100, 150, 220, 300):
        out.append(("kmeans", kmeans_placement(inst, rng, altitude=alt)))
    for _ in range(9):
        u = kmeans_placement(inst, rng, altitude=rng.uniform(50, 300))
        u += rng.standard_normal(u.shape) * [150, 150, 30]
        out.append(("kmeans-perturbed", clip(inst, u)))
    # snapshots along a local search: 60, 250, 1000 and 4000 scorer calls, from three starts
    for _ in range(3):
        for budget in (60, 250, 1000, 4000):
            sub = np.random.default_rng(int(rng.integers(1 << 30)))
            out.append((f"search-{budget}", local_search(inst, sub, budget=budget)))
    return out


def main(parallel: int):
    done = set()
    if OUT.exists():
        done = {(r["instance"], r["index"]) for r in map(json.loads, OUT.read_text().splitlines())}
    jobs = []
    for cfg in CONFIGS:
        inst = ub.generate(**cfg)
        rng = np.random.default_rng(cfg["seed"])
        for i, (kind, u) in enumerate(placements(inst, rng)):
            if (inst.id, i) not in done:
                jobs.append((inst, i, kind, u))

    def run(job):
        inst, i, kind, u = job
        f = ub.score_fast(inst, u, detail=True)
        n = ub.score_ns3(inst, u, detail=True)
        return dict(instance=inst.id, index=i, kind=kind, uavs=np.round(u, 2).tolist(),
                    fast=f["score"], ns3=n["score"], fast_sum=f["sum_thr_mbps"],
                    ns3_sum=n["sum_thr_mbps"], wall_s=n["wall_s"],
                    user_corr=float(np.corrcoef(f["thr_mbps"], n["thr_mbps"])[0, 1]))

    with ThreadPoolExecutor(parallel) as ex, OUT.open("a") as fh:
        for r in ex.map(run, jobs):
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            print(r["instance"], r["index"], r["kind"], round(r["fast"], 3), round(r["ns3"], 3),
                  flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 4)
