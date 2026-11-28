"""Does the ns-3 scorer give the same scores on this machine as where they were first measured?

  python experiments/ns3_machine_check.py <set> [workers]     e.g. public, public-cincinnati-downtown
Rescores every answer of results/compare-<set>.jsonl with ns-3 here and compares with the ns-3
score stored there (and the closed-form score, recomputed). Writes
results/ns3-machine-check-<set>.jsonl and prints the largest differences.
"""
import json
import os
import platform
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import skycellbench as ub

ROOT = Path(__file__).resolve().parent.parent


def rescore(row):
    inst = ub.Instance.load(ROOT / "instances" / sys.argv[1] / f"{row['instance']}.json")
    return dict(instance=row["instance"], controller=row["controller"],
                fast_then=row["fast"], fast_here=ub.score_fast(inst, row["uavs"])["score"],
                ns3_then=row["ns3"], ns3_here=ub.score_ns3(inst, row["uavs"])["score"],
                host=platform.node())


if __name__ == "__main__":
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    rows = [r for r in map(json.loads, (ROOT / "experiments" / "results" /
                                        f"compare-{name}.jsonl").open()) if "ns3" in r]
    with Pool(workers) as pool:
        out = pool.map(rescore, rows)
    (ROOT / "experiments" / "results" / f"ns3-machine-check-{name}.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in out))
    for k in ("fast", "ns3"):
        d = np.array([r[f"{k}_here"] - r[f"{k}_then"] for r in out])
        print(f"{name} {k}: {len(d)} answers, identical {int((d == 0).sum())}, "
              f"largest difference {np.abs(d).max():.2e}, mean {d.mean():+.2e}")
