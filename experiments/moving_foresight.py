"""What is knowing the future worth under a speed limit, over reacting?

  PYTHONPATH=. python experiments/moving_foresight.py [calls]
experiments/moving_react.py found reacting plans that score as high as the plans of
moving_limit.py, which were made with every epoch known, and in places higher than its plans
with no limit at all: that search had not settled. So the comparison is made again from the
best plans known. For every instance on the scene, with the positions of all plans found so far
as candidates, the search of moving_limit.py (which knows every epoch) is run for two more rounds
  fixed      from the fixed plan
  planned    under each limit of moving_react.py, from the best plan known under that limit:
             the plan of moving_limit.py or one of the two reacting plans
  free       with no limit, from the best plan known in each epoch
`planned` minus the better reacting plan is what the search could add by knowing the future: a
lower bound on what foresight is worth. Closed-form scores. Appends to
results/moving-foresight.jsonl and skips what is already there.
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moving_limit as ml
import moving_react as mr
import moving_scene as ms
import skycellbench as ub

OUT = Path(__file__).resolve().parent / "results" / "moving-foresight.jsonl"


class Search(ml.Search):
    """With the positions of every known plan there are too many candidates to score them all for
    every UAV: each path step takes the 125 nearest to the UAV's own path and 125 at random."""

    def candidates(self, plan, k, limit, free):
        full = self.seeds
        if len(full) > 250:
            d = np.linalg.norm(full[:, None] - plan[None, :, k], axis=-1).min(axis=1)
            near = np.argsort(d)[:125]
            rest = np.setdiff1d(np.arange(len(full)), near)
            self.seeds = full[np.r_[near, self.rng.choice(rest, 125, replace=False)]]
        try:
            return super().candidates(plan, k, limit, free)
        finally:
            self.seeds = full


def run(job):
    kind, seed, k, n, calls = job
    inst = ms.moving_instance(kind, seed, k, n, None)
    swept = {r["limit"]: r for r in map(json.loads, ml.OUT.open())
             if r["inst"] == inst.id and r["channel"] == "scene"}
    react = {r["limit"]: r for r in map(json.loads, mr.OUT.open())
             if r["inst"] == inst.id and r["channel"] == "scene"}
    plans = {("swept", l): np.array(r["plan"]) for l, r in swept.items()}
    for l, r in react.items():
        plans[("react", l)] = np.array(r["react_plan"])
        plans[("react_home", l)] = np.array(r["react_home_plan"])
    rng = np.random.default_rng([seed, 17])
    se = Search(inst, list(plans.values()), rng)
    # stored plans are rounded to centimetres; keep those that are still valid
    plans = {key: p for key, p in plans.items()
             if all(se.valid(p[:, j], key[1]) for j in range(k))}
    score = {key: float(se.w @ se.epochs(p)) for key, p in plans.items()}
    evals = calls * se.E
    # free: the best known plan of each epoch, then the search with no limit
    table = np.array([se.epochs(p) for p in plans.values()])
    free = np.array([list(plans.values())[i][e] for e, i in enumerate(table.argmax(axis=0))])
    free_start = float(se.w @ se.epochs(free))
    free = ml.relabel(se.improve(free, None, free, ml.ROUNDS, evals), free[0])
    fixed = se.improve(plans[("swept", 0.0)], 0.0, free, ml.ROUNDS, evals)
    row = dict(inst=inst.id, kind=kind, calls=calls,
               fixed_start=round(score[("swept", 0.0)], 4),
               fixed=round(ub.score_fast(inst, fixed)["score"], 4),
               free_start=round(free_start, 4), free=round(ub.score_fast(inst, free)["score"], 4),
               limits=[])
    for limit in sorted(react):
        known = {key: s for key, s in score.items() if key[1] == limit}
        start = max(known, key=known.get)
        plan = se.improve(plans[start], limit, ml.relabel(free, plans[start][0]), ml.ROUNDS, evals)
        ub.check(dataclasses.replace(inst, max_move_m=limit), plan)
        row["limits"].append(dict(
            limit=limit, swept=round(score.get(("swept", limit), np.nan), 4),
            react=round(score.get(("react", limit), np.nan), 4),
            react_home=round(score.get(("react_home", limit), np.nan), 4), start=start[0],
            planned=round(ub.score_fast(inst, plan)["score"], 4), plan=np.round(plan, 2).tolist()))
    return row


if __name__ == "__main__":
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    only = sys.argv[2:]
    have = {r["inst"] for r in map(json.loads, mr.OUT.open()) if r["channel"] == "scene"}
    done = {r["inst"] for r in map(json.loads, OUT.open())} if OUT.exists() else set()
    jobs = [(kind, seed, k, n, calls) for kind, seed, k, n in ml.CASES
            if f"moving-{kind}-{seed}-k{k}-n{n}" in have - done
            and (not only or f"moving-{kind}-{seed}-k{k}-n{n}" in only)]
    jobs.sort(key=lambda j: -j[2] * j[3])
    with Pool(4) as pool, OUT.open("a") as f:
        for row in pool.imap_unordered(run, jobs):
            f.write(json.dumps(row) + "\n")
            f.flush()
            print({k: v for k, v in row.items() if k != "limits"},
                  [{k: v for k, v in l.items() if k != "plan"} for l in row["limits"]], flush=True)
