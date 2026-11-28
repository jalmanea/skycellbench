"""Run every entered controller on one instance set and score each answer with both scorers, as
PROTOCOL.md (sections 4 to 6) says.

  python experiments/heldout.py <set> [workers]        e.g. heldout, heldout-cincinnati-downtown
Writes results/heldout/<set>.jsonl, one row per answer, and skips answers already there, so a
stopped run continues. experiments/analyse_heldout.py makes the tables.

What is run, per instance:
  low track    centre, random (seeds 0-2), kmeans (not blind), search30 (seeds 0-2), and the RL
               entry: rl on flat ground, rl_scene on a scene
  high track   search, 4,000 calls (seeds 0-2); not on blind
  transfer     on a scene, rl as well (the policy trained on flat ground); reported beside the
               entry, not as one
Deterministic controllers run once (seed 0). Each answer records the scorer calls it used:
calls to the closed-form scorer and to the RL environment's feedback on place and constrained,
probes on blind. A controller that used more than its track allows, or returned an answer that
breaks a rule, is marked and scores 0.

On a held-out set the run refuses to start unless the working tree is clean apart from
experiments/results/, so that the commit recorded with every row is what was run. Any other set (a public one, as a trial) runs
from any tree.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench import baselines, controllers, fast, rl

ROOT = Path(__file__).resolve().parent.parent
BUDGET = {"low": 30, "high": 4000}
RANDOM = {"random", "search30", "search"}


def git(*args) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def plan(inst: ub.Instance) -> list[tuple[str, str, int]]:
    """(controller, track, seed) for every answer to make on this instance."""
    low = ["centre", "random"] + ([] if inst.task == "blind" else ["kmeans"]) + ["search30"]
    low.append("rl_scene" if inst.scene else "rl")
    out = [(c, "low", s) for c in low for s in (range(3) if c in RANDOM else [0])]
    if inst.task != "blind":
        out += [("search", "high", s) for s in range(3)]
    if inst.scene:
        out.append(("rl", "transfer", 0))
    return out


class Count:
    """Counts calls to a scorer function in the modules that use it."""

    def __init__(self):
        self.n = 0

    def wrap(self, fn):
        def counted(*a, **k):
            self.n += 1
            return fn(*a, **k)
        return counted


def answer(job) -> dict:
    path, cname, track, seed = job
    inst = ub.Instance.load(path)
    count = Count()
    controllers.score_fast = count.wrap(fast.score_fast)
    baselines.score_fast = controllers.score_fast
    rl.feedback = count.wrap(fast.feedback)
    probe = ub.Probe(inst) if inst.task == "blind" else None
    rng = np.random.default_rng([inst.seed, len(cname), seed])
    t0 = time.time()
    u = controllers.ALL[cname](inst.public(), probe, rng)
    wall = time.time() - t0
    calls = count.n if probe is None else inst.probe_budget - probe.remaining
    row = dict(instance=inst.id, task=inst.task, scene=inst.scene, controller=cname,
               track=track, seed=seed, calls=calls, controller_wall_s=round(wall, 2))
    limit = BUDGET["low" if track == "transfer" else track]
    problem = None
    if calls > limit:
        problem = f"used {calls} scorer calls, {limit} allowed"
    else:
        try:
            u = ub.check(inst, u)
        except Exception as e:                         # an answer that breaks a rule
            problem = f"invalid: {e}"
    if problem:
        row.update(valid=False, problem=problem, uavs=None, fast=0.0, ns3=0.0)
        return row
    demand = inst.demands(0)
    f = ub.score_fast(inst, u, detail=True)
    n = ub.score_ns3(inst, u, detail=True)
    sat = np.minimum(1.0, np.asarray(n["thr_mbps"]) / demand)
    row.update(valid=True, uavs=np.round(u, 2).tolist(),
               fast=f["score"], fast_sum_thr_mbps=f["sum_thr_mbps"],
               fast_p5=f["p5_satisfaction"],
               ns3=n["score"], ns3_sum_thr_mbps=n["sum_thr_mbps"],
               ns3_p5=float(np.percentile(sat, 5)), ns3_wall_s=n["wall_s"])
    return row


def main():
    name = sys.argv[1]
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else os.cpu_count()
    files = sorted((ROOT / "instances" / name).glob("*.json"))
    assert files, f"no instances in instances/{name}"
    # results written by this or another run of the same commit do not make the tree unclean
    dirty = "\n".join(l for l in git("status", "--porcelain").splitlines()
                      if not l[3:].startswith("experiments/results/"))
    if name.startswith("heldout") and dirty:
        sys.exit("held-out run refused: the working tree has uncommitted changes:\n" + dirty)
    out = ROOT / "experiments" / "results" / "heldout" / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["instance"], r["controller"], r["track"], r["seed"])
                for r in map(json.loads, out.open())}
    run = dict(commit=git("rev-parse", "HEAD"),
               protocol_commit=git("log", "-1", "--format=%H", "--", "PROTOCOL.md"),
               tree_clean=not dirty, host=platform.node(), set=name)
    jobs = []
    for p in files:
        inst = ub.Instance.load(p)
        jobs += [(p, c, t, s) for c, t, s in plan(inst) if (inst.id, c, t, s) not in done]
    # the 4,000-call searches first, so that the slow jobs do not come last
    jobs.sort(key=lambda j: j[2] != "high")
    print(f"{name}: {len(done)} answers done, {len(jobs)} to make, {workers} workers, "
          f"commit {run['commit'][:10]}", flush=True)
    with Pool(workers, maxtasksperchild=20) as pool, out.open("a") as fh:
        for i, row in enumerate(pool.imap_unordered(answer, jobs)):
            row.update(run)
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            if (i + 1) % 25 == 0 or i + 1 == len(jobs):
                print(f"{i + 1}/{len(jobs)} {row['instance']} {row['controller']} "
                      f"ns-3 {row['ns3']:.3f}", flush=True)


if __name__ == "__main__":
    main()
