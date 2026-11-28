"""Does it take knowledge of the future to gain from moving under a speed limit?

  PYTHONPATH=. python experiments/moving_react.py [calls] [scene | statistical]
experiments/moving_limit.py found plans under a speed limit that beat fixed UAVs, with a search
that knows where the users will be in every epoch. Here the UAVs react instead: in every epoch
they take the best positions for the users of that epoch that the limit allows, and nothing of
later epochs is used. Two starts:
  react        epoch 0 is the best placement for the users of epoch 0
  react_home   epoch 0 lies within the limit of the best fixed positions of moving_limit.py, a
               home chosen with every epoch known (an operator who knows where demand usually is)
Each epoch is searched like a plan in moving_limit.py: one UAV at a time, the best of a set of
candidate positions within reach (its own, known good positions, random points in reach, and the
point at the limit on the way to each known good position), three times over, then `2 * calls`
random moves kept when the score rises. The scores of the fixed, planned and free plans are read
from results/moving-limit.jsonl; instances not yet there are skipped.

Closed-form scores. Appends to results/moving-react.jsonl and skips what is already there.
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import moving_limit as ml
import moving_scene as ms
import skycellbench as ub

OUT = Path(__file__).resolve().parent / "results" / "moving-react.jsonl"
LIMITS = (100.0, 150.0, 300.0, 600.0)


def best_near(se: ml.Search, e: int, prev, limit: float, start, evals: int) -> np.ndarray:
    """The best positions found for epoch e with every UAV within `limit` of `prev` (None: any)."""
    rng = se.rng

    def near(p, k):
        if prev is None:
            return np.ones(np.shape(p)[:-1], bool)
        return np.linalg.norm(p - prev[k], axis=-1) <= limit

    u = start.copy()
    s = se.epoch(e, u)
    for _ in range(3):
        for k in rng.permutation(se.K):
            pts = [se.seeds]
            if prev is not None:
                d = se.seeds - prev[k]
                far = np.linalg.norm(d, axis=-1)
                pts.append(prev[k] + d[far > limit] * (0.98 * limit / far[far > limit])[:, None])
                r = rng.standard_normal((40, 3))
                r *= (limit * rng.random(40) ** (1 / 3) / np.linalg.norm(r, axis=-1))[:, None]
                pts.append(prev[k] + r)
            else:
                pts += [u[k] + rng.normal(0, (g, g, 30.0), size=(15, 3)) for g in (30.0, 100.0, 300.0)]
            cand = se.clean(np.vstack(pts))
            cand = cand[near(cand, k)]
            v = u.copy()
            for c in cand:
                v[k] = c
                t = se.epoch(e, v)
                if t > s:
                    u, s = v.copy(), t
    scale = 0.5 * (se.hi - se.lo)
    step, used, tries = 0.05, 0, 0
    while used < evals and tries < 20 * evals:
        tries += 1
        k = rng.integers(se.K)
        v = u.copy()
        v[k] = np.clip(v[k] + step * scale * rng.standard_normal(3), se.lo, se.hi)
        if not se.valid(v[k:k + 1], None) or not near(v[k], k):
            step = max(step * 0.95, 2e-3)
            continue
        t = se.epoch(e, v)
        used += 1
        if t > s:
            u, s, step = v, t, min(step * 1.5, 0.3)
        else:
            step *= 0.9
            if step < 2e-3:
                step = 0.05
    return u


def run(job):
    kind, seed, k, n, channel, calls = job
    inst = ms.moving_instance(kind, seed, k, n, None)
    if channel == "statistical":
        inst = dataclasses.replace(inst, scene=None)
    swept = {r["limit"]: r for r in map(json.loads, ml.OUT.open())
             if r["inst"] == inst.id and r["channel"] == channel}
    rng = np.random.default_rng([seed, 13])
    se = ml.Search(inst, [swept[l]["plan"] for l in swept], rng)
    home, free = np.array(swept[0.0]["plan"])[0], np.array(swept[None]["plan"])
    home, ok = se.lift(home)
    assert ok.all()
    rows = []
    for limit in LIMITS:
        row = dict(inst=inst.id, kind=kind, channel=channel, calls=calls, limit=limit,
                   fixed=swept[0.0]["score"], planned=swept[limit]["score"],
                   free=swept[None]["score"])
        for name in ("react", "react_home"):
            plan = []
            for e in range(se.E):
                if e == 0 and name == "react":
                    first, ok = se.lift(free[0])
                    u = best_near(se, 0, None, limit, np.where(ok[:, None], first, home), 2 * calls)
                else:
                    prev = plan[-1] if plan else home
                    u = best_near(se, e, prev, limit, prev, 2 * calls)
                plan.append(u)
            plan = np.array(plan)
            ub.check(dataclasses.replace(inst, max_move_m=limit), plan)
            row[name] = round(ub.score_fast(inst, plan)["score"], 4)
            row[name + "_epochs"] = np.round(se.epochs(plan), 4).tolist()
            row[name + "_plan"] = np.round(plan, 2).tolist()
        row["planned_epochs"], row["fixed_epochs"] = swept[limit]["epochs"], swept[0.0]["epochs"]
        rows.append(row)
    return rows


if __name__ == "__main__":
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    swept = {(r["inst"], r["channel"]) for r in map(json.loads, ml.OUT.open())
             if r["limit"] is None}
    done = set()
    if OUT.exists():
        done = {(r["inst"], r["channel"], r["calls"]) for r in map(json.loads, OUT.open())}
    jobs = [(kind, seed, k, n, channel, calls)
            for channel in (sys.argv[2:] or ("scene", "statistical"))
            for kind, seed, k, n in ml.CASES
            if (f"moving-{kind}-{seed}-k{k}-n{n}", channel) in swept
            and (f"moving-{kind}-{seed}-k{k}-n{n}", channel, calls) not in done]
    jobs.sort(key=lambda j: (j[4] != "scene", -j[2] * j[3]))
    with Pool(4) as pool, OUT.open("a") as f:
        for rows in pool.imap_unordered(run, jobs):
            for row in rows:
                f.write(json.dumps(row) + "\n")
                print({k: v for k, v in row.items() if not k.endswith(("plan", "epochs"))},
                      flush=True)
            f.flush()
