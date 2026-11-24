"""How much of the value of moving is left under a speed limit? A sweep over the limit.

  PYTHONPATH=. python experiments/moving_limit.py [calls]
experiments/moving_scene.py found that UAVs free to jump anywhere each epoch beat fixed UAVs on
the scene, and that its searches gained almost nothing under limits of 40 to 150 m per epoch. That
left two explanations: the limits were too tight, or the searches too weak. Here one and the same
search is run under every limit, from 0 (fixed UAVs) to none (free), on the same instances.

The search, for one limit:
  path step   one UAV at a time, the others held: every candidate position is scored in every
              epoch, and the best path through the candidates that keeps to the limit is found
              exactly (dynamic programme over the epochs). Candidates are the UAV's own positions,
              all positions of the fixed and free plans of moving_scene.py, random points around
              its own positions, and points one limit apart on the way from its own positions to
              those of the free plan.
  polish      random moves of one UAV in one epoch, a run of epochs or all epochs, kept when the
              score rises; `calls` scores of a whole plan's worth.
Two rounds of (path step for every UAV, polish) per limit. The limits are run in rising order and
each starts from the answer of the one before, which stays valid, so the curve cannot fall. The
fixed plan gets six rounds, and its score after two is kept to show that it has settled.

Closed-form scores. Appends to results/moving-limit.jsonl and skips what is already there.
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

import moving_scene as ms
import skycellbench as ub
from skycellbench.baselines import ROOF_CLEARANCE_M
from skycellbench.fast import _snapshot
from skycellbench.instance import TOLERANCE_M

OUT = Path(__file__).resolve().parent / "results" / "moving-limit.jsonl"
LIMITS = (0.0, 50.0, 100.0, 150.0, 300.0, 600.0, 1000.0, None)        # metres per epoch
CASES = [c[:4] for c in ms.CASES if c[0] != "events"]
ROUNDS, ROUNDS_FIXED = 2, 6


class Search:
    def __init__(self, inst: ub.Instance, seeds, rng):
        assert inst.task == "place" and inst.max_move_m is None
        self.inst, self.rng = inst, rng
        self.sc = inst.get_scene()
        self.E, self.K = inst.n_epochs, inst.n_uav
        self.xy = [inst.xy(e) for e in range(self.E)]
        self.demand = [inst.demands(e) for e in range(self.E)]
        self.w = np.array([inst.active(e).sum() for e in range(self.E)], float)
        self.w /= self.w.sum()
        self.lo = np.array([0, 0, inst.alt_min_m])
        self.hi = np.array([inst.side_m, inst.side_m, inst.alt_max_m])
        self.seeds = self.clean(np.vstack([np.reshape(s, (-1, 3)) for s in seeds]))

    def epoch(self, e, u) -> float:
        return _snapshot(self.inst, u, self.xy[e], self.demand[e], False)["score"]

    def epochs(self, plan) -> np.ndarray:
        return np.array([self.epoch(e, plan[e]) for e in range(self.E)])

    def lift(self, pts):
        """Points moved inside the area and above any roof, and which of them are then valid."""
        p = np.clip(np.asarray(pts, float).reshape(-1, 3), self.lo, self.hi)
        if self.sc is None:
            return p, np.ones(len(p), bool)
        roof = self.sc.building_height(p[:, :2])
        low = p[:, 2] <= roof
        p[low, 2] = roof[low] + ROOF_CLEARANCE_M
        return p, p[:, 2] <= self.inst.alt_max_m

    def clean(self, pts) -> np.ndarray:
        """The valid ones of these points, each once."""
        p, ok = self.lift(pts)
        return np.unique(np.round(p[ok], 2), axis=0)

    def valid(self, path, limit) -> bool:
        if self.sc is not None and (path[:, 2] <= self.sc.building_height(path[:, :2])).any():
            return False
        return limit is None or len(path) < 2 or \
            np.linalg.norm(np.diff(path, axis=0), axis=-1).max() <= limit + TOLERANCE_M

    def candidates(self, plan, k, limit, free) -> np.ndarray:
        cur = plan[:, k]
        pts = [self.seeds] + [cur + self.rng.normal(0, (s, s, 30.0), size=cur.shape)
                              for s in (30.0, 100.0, 300.0)]
        if limit:                                    # stepping stones towards the free plan
            ends = np.vstack([free[:, k], free[1:, k]])
            starts = np.vstack([cur, free[:-1, k]])
            for a, b in zip(starts, ends):
                d = np.linalg.norm(b - a)
                steps = np.arange(1, min(int(d / limit), 6) + 1) * 0.98 * limit
                pts.append(a + (b - a) * steps[:, None] / max(d, 1e-9))
        return np.vstack([np.unique(cur, axis=0), self.clean(np.vstack(pts))])

    def path_step(self, plan, k, limit, free) -> np.ndarray:
        """The best path for UAV k through the candidates, the other UAVs held."""
        cand = self.candidates(plan, k, limit, free)
        val = np.empty((self.E, len(cand)))
        for e in range(self.E):
            u = plan[e].copy()
            for m, c in enumerate(cand):
                u[k] = c
                val[e, m] = self.w[e] * self.epoch(e, u)
        if limit is None:
            return cand[val.argmax(axis=1)]
        ok = np.linalg.norm(cand[:, None] - cand[None], axis=-1) <= limit + TOLERANCE_M
        best, back = val[0].copy(), []
        for e in range(1, self.E):
            reach = np.where(ok, best[:, None], -np.inf)          # [from, to]
            arg = reach.argmax(axis=0)
            back.append(arg)
            best = val[e] + reach[arg, np.arange(len(cand))]
        m = int(best.argmax())
        path = [m]
        for arg in reversed(back):
            m = int(arg[m])
            path.append(m)
        return cand[path[::-1]]

    def polish(self, plan, limit, evals) -> np.ndarray:
        """Random moves kept when the score rises; `evals` counts scores of one epoch."""
        s = self.epochs(plan)
        scale = 0.5 * (self.hi - self.lo)
        step, used, tries = 0.05, self.E, 0
        while used < evals and tries < 20 * evals:
            tries += 1
            k = self.rng.integers(self.K)
            a, b = 0, self.E
            if limit != 0 and self.rng.random() < 0.8:
                a = int(self.rng.integers(self.E))
                b = a + 1 + (int(self.rng.integers(self.E - a)) if self.rng.random() < 0.5 else 0)
            v = plan.copy()
            v[a:b, k] = np.clip(v[a:b, k] + step * scale * self.rng.standard_normal(3),
                                self.lo, self.hi)
            if not self.valid(v[:, k], limit):
                step = max(step * 0.95, 2e-3)
                continue
            t = s.copy()
            t[a:b] = [self.epoch(e, v[e]) for e in range(a, b)]
            used += b - a
            if self.w @ t > self.w @ s:
                plan, s, step = v, t, min(step * 1.5, 0.3)
            else:
                step *= 0.9
                if step < 2e-3:
                    step = 0.05
        return plan

    def improve(self, plan, limit, free, rounds, evals, trace=None) -> np.ndarray:
        for _ in range(rounds):
            for k in self.rng.permutation(self.K):
                new = plan.copy()
                new[:, k] = self.path_step(plan, k, limit, free)
                if self.w @ self.epochs(new) >= self.w @ self.epochs(plan):
                    plan = new
            plan = self.polish(plan, limit, evals)
            if trace is not None:
                trace.append(round(float(self.w @ self.epochs(plan)), 4))
        return plan


def relabel(plan, first) -> np.ndarray:
    """The same plan with the UAVs renumbered so that they move as little as possible."""
    out, prev = [], first
    for u in plan:
        cost = np.linalg.norm(prev[:, None] - u[None], axis=-1)
        u = u[linear_sum_assignment(cost)[1]]
        out.append(u)
        prev = u
    return np.array(out)


def run(job):
    kind, seed, k, n, channel, calls = job
    inst = ms.moving_instance(kind, seed, k, n, None)
    if channel == "statistical":
        inst = dataclasses.replace(inst, scene=None)
    old = next(r for r in map(json.loads, ms.OUT.open())
               if r["inst"] == inst.id and r["channel"] == channel)
    rng = np.random.default_rng([seed, 11])
    fixed0, free0 = np.array(old["plans"]["static"]), np.array(old["plans"]["free"])
    se = Search(inst, [fixed0, free0], rng)
    home, ok = se.lift(fixed0[0])      # stored plans are rounded; lift what that put in a roof
    assert ok.all()
    plan = np.repeat(home[None], se.E, axis=0)
    free = relabel(np.array([np.where(ok[:, None], p, home) for p, ok in map(se.lift, free0)]),
                   home)
    evals = calls * se.E
    rows, trace = [], []
    for limit in LIMITS:
        if limit is None:            # each epoch alone: take the better of the two plans in each
            better = se.epochs(free) > se.epochs(plan)
            plan = np.where(better[:, None, None], free, plan)
        elif limit and all(se.valid(free[:, j], limit) for j in range(k)) \
                and se.w @ se.epochs(free) > se.w @ se.epochs(plan):
            plan = free.copy()       # the free plan itself keeps to this limit
        plan = se.improve(plan, limit, free, ROUNDS_FIXED if limit == 0 else ROUNDS, evals,
                          trace if limit == 0 else None)
        ub.check(dataclasses.replace(inst, max_move_m=limit), plan)
        moved = np.linalg.norm(np.diff(plan, axis=0), axis=-1)
        rows.append(dict(inst=inst.id, kind=kind, channel=channel, calls=calls,
                         limit=limit, score=round(ub.score_fast(inst, plan)["score"], 4),
                         epochs=np.round(se.epochs(plan), 4).tolist(),
                         step_mean=round(float(moved.mean()), 1),
                         step_max=round(float(moved.max()), 1),
                         fixed_by_round=trace if limit == 0 else None,
                         old_static=old["static"], old_free=old["free"],
                         plan=np.round(plan, 2).tolist()))
        if limit is None:
            f = relabel(plan, plan[0])
            m = np.linalg.norm(np.diff(f, axis=0), axis=-1)
            rows[-1].update(free_step_median=round(float(np.median(m)), 1),
                            free_step_p90=round(float(np.percentile(m, 90)), 1),
                            free_step_max=round(float(m.max()), 1))
    return rows


if __name__ == "__main__":
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    only = sys.argv[2:]                                        # instance ids, for a short trial
    done = set()
    if OUT.exists():
        done = {(r["inst"], r["channel"], r["calls"]) for r in map(json.loads, OUT.open())}
    jobs = [(kind, seed, k, n, channel, calls)
            for channel in ("scene", "statistical") for kind, seed, k, n in CASES
            if (f"moving-{kind}-{seed}-k{k}-n{n}", channel, calls) not in done
            and (not only or f"moving-{kind}-{seed}-k{k}-n{n}" in only)]
    jobs.sort(key=lambda j: (j[4] != "scene", -j[2] * j[3]))   # the slow ones first
    with Pool(4) as pool, OUT.open("a") as f:
        for rows in pool.imap_unordered(run, jobs):
            for row in rows:
                f.write(json.dumps(row) + "\n")
                print({k: v for k, v in row.items() if k not in ("plan", "epochs")}, flush=True)
            f.flush()
