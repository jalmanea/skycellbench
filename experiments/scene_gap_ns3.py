"""The gap check of scene_gap.py, scored again with ns-3.

  PYTHONPATH=. python experiments/scene_gap_ns3.py <set> [workers]
Takes the k-means answer and the reference answer of every instance from
results/scene-gap-<set>.jsonl, on the scene and on the old channel, and scores each with ns-3
(about 20 s each). The answers are the stored ones, rounded to centimetres; an answer that
rounding made invalid is reported and left out. Appends to results/scene-gap-ns3-<set>.jsonl
and skips what is already there; `summary` in place of the workers prints the table.
"""
import dataclasses
import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import skycellbench as ub

root = Path(__file__).resolve().parent
name = sys.argv[1]
src = root / "results" / f"scene-gap-{name}.jsonl"
out = root / "results" / f"scene-gap-ns3-{name}.jsonl"


def run(job):
    r, kind = job
    inst = ub.Instance.load(root.parent / "instances" / name / f"{r['instance']}.json")
    if r["channel"] == "statistical":
        inst = dataclasses.replace(inst, scene=None)
    row = dict(instance=r["instance"], task=r["task"], channel=r["channel"], answer=kind,
               calls=r["calls"])
    try:
        u = np.array(r[f"{kind}_uavs"], float)
        row.update(fast=ub.score_fast(inst, u)["score"], ns3=ub.score_ns3(inst, u)["score"])
    except ValueError as e:
        row["error"] = str(e)
    return row


def summary():
    by = defaultdict(dict)
    for r in map(json.loads, out.open()):
        if "ns3" in r:
            by[(r["task"], r["channel"], r["instance"])][r["answer"]] = r
    print("k-means against the reference search, both scorers; gap = reference minus k-means")
    for task in sorted({t for t, _, _ in by}):
        print(f"\n{task}")
        gaps = {}
        for channel in ("scene", "statistical"):
            d = {i: v for (t, c, i), v in by.items() if t == task and c == channel and len(v) == 2}
            for scorer in ("ns3", "fast"):
                km = np.array([v["kmeans"][scorer] for _, v in sorted(d.items())])
                ref = np.array([v["reference"][scorer] for _, v in sorted(d.items())])
                gaps[(channel, scorer)] = dict(zip(sorted(d), ref - km))
                print(f"  {'scene' if channel == 'scene' else 'old channel':<12}"
                      f"{'ns-3' if scorer == 'ns3' else 'closed-form':<12} {len(d)} instances  "
                      f"k-means {km.mean():.3f}  reference {ref.mean():.3f}  "
                      f"gap {(ref - km).mean():.3f} (min {(ref - km).min():.3f})")
        for scorer in ("ns3", "fast"):
            a, b = gaps[("scene", scorer)], gaps[("statistical", scorer)]
            both = sorted(set(a) & set(b))
            diff = np.array([a[i] - b[i] for i in both])
            print(f"  gap on the scene minus gap on the old channel, "
                  f"{'ns-3' if scorer == 'ns3' else 'closed-form'}: mean {diff.mean():+.3f}, "
                  f"larger on the scene in {int((diff > 0).sum())} of {len(both)}")


if __name__ == "__main__":
    if sys.argv[2:] == ["summary"]:
        summary()
        sys.exit()
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    rows = [json.loads(l) for l in src.open()]
    calls = max(r["calls"] for r in rows)
    done = set()
    if out.exists():
        done = {(r["instance"], r["channel"], r["answer"]) for r in map(json.loads, out.open())}
    jobs = [(r, kind) for r in rows if r["calls"] == calls for kind in ("kmeans", "reference")
            if (r["instance"], r["channel"], kind) not in done]
    with ThreadPoolExecutor(workers) as ex, out.open("a") as f:
        for row in ex.map(run, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print(row, flush=True)
