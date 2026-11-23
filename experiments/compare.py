"""Run every controller on an instance set and score the answers.

  python experiments/compare.py public            closed-form scores only
  python experiments/compare.py public --ns3      also ns-3 (about 20 s per answer)
Appends to results/compare-<set>.jsonl and skips what is already there.
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench import controllers

root = Path(__file__).resolve().parent
name, use_ns3 = sys.argv[1], "--ns3" in sys.argv
out = root / "results" / f"compare-{name}.jsonl"
done = {}
if out.exists():
    done = {(r["instance"], r["controller"]): r for r in map(json.loads, out.open())}

jobs = []
for path in sorted((root.parent / "instances" / name).glob("*.json")):
    inst = ub.Instance.load(path)
    for cname, fn in controllers.ALL.items():
        row = done.get((inst.id, cname))
        if row is None:
            probe = ub.Probe(inst) if inst.task == "blind" else None
            u = fn(inst.public(), probe, np.random.default_rng([inst.seed, len(cname)]))
            if u is None:
                continue
            row = dict(instance=inst.id, task=inst.task, controller=cname,
                       uavs=np.round(u, 2).tolist(), fast=ub.score_fast(inst, u)["score"])
        if use_ns3 and "ns3" not in row:
            jobs.append((inst, row))
        done[(inst.id, cname)] = row


def flush():
    out.write_text("".join(json.dumps(r) + "\n" for r in done.values()))


flush()
print(len(done), "answers;", len(jobs), "to score with ns-3", flush=True)


def run(job):
    inst, row = job
    row["ns3"] = ub.score_ns3(inst, row["uavs"])["score"]
    return row


with ThreadPoolExecutor(4) as ex:
    for i, row in enumerate(ex.map(run, jobs)):
        flush()
        print(i + 1, row["instance"], row["controller"], round(row["fast"], 3),
              round(row["ns3"], 3), flush=True)
