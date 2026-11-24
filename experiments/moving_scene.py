"""Does moving the UAVs pay on a real scene? The test of results/moving-lookahead.txt, repeated.

  PYTHONPATH=. python experiments/moving_scene.py [calls]
Users move over several epochs on the scene and the UAVs may move at most `mm` metres per epoch.
Four kinds of plan are scored with the closed-form scorer:
  static   one position per UAV for all epochs
  free     the best position in every epoch, with no speed limit
  myopic   each epoch optimised alone, within `mm` of the positions of the epoch before
  planner  the whole trajectory optimised under the limit
Every search is a local search with `calls` scorer calls: per epoch for free and myopic, and three
times as many for the planner, which has a whole trajectory to find. The free, myopic and planner
searches start from the static answer, so they can only do better than it; what is reported is
how much better. Each instance is also run on the old statistical
channel with the same users (the scene removed).

Three kinds of instance. No task uses any of them: this script is the generator.
  drift    hotspots drift across the area and some start late or end early; 8 epochs
  events   short events at two to three sites per UAV; 12 epochs. Too light a load: static UAVs
           already deliver everything, so it shows nothing
  events3  the same events with three times the demand per user
Appends to results/moving-scene.jsonl and skips what is already there.
"""
import dataclasses
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench.baselines import local_search

SCENE = "cincinnati-downtown"
OUT = Path(__file__).resolve().parent / "results" / "moving-scene.jsonl"
CASES = [("drift", 1, 3, 150, (100.0, 40.0)), ("drift", 2, 5, 200, (100.0, 40.0)),
         ("drift", 3, 8, 250, (100.0, 40.0)), ("drift", 4, 5, 150, (100.0, 40.0)),
         ("events", 1, 3, 150, (150.0, 75.0)), ("events", 2, 5, 200, (150.0, 75.0)),
         ("events", 3, 8, 250, (150.0, 75.0)), ("events", 4, 5, 150, (150.0, 75.0)),
         ("events3", 1, 3, 150, (150.0, 75.0)), ("events3", 2, 5, 200, (150.0, 75.0)),
         ("events3", 3, 8, 250, (150.0, 75.0)), ("events3", 4, 5, 150, (150.0, 75.0))]


def moving_instance(kind: str, seed: int, n_uav: int, n_users: int, max_move_m: float,
                    scene: str = SCENE) -> ub.Instance:
    """Users on the scene who move from epoch to epoch and are active only part of the time."""
    sc = ub.get_scene(scene)
    side = sc.size[0]
    rng = np.random.default_rng([seed, 2, int(kind != "drift")])
    demand = rng.choice((0.5, 1.0, 2.0), n_users, p=(0.4, 0.4, 0.2))
    if kind == "events3":
        demand = 3 * demand
    if kind == "drift":
        epochs, n_hot = 8, int(rng.integers(2, 6))
        hot = np.where(rng.random(n_users) < rng.uniform(0.3, 0.8),
                       rng.choice(n_hot, n_users, p=rng.dirichlet(np.full(n_hot, 2.0))), -1)
        speed = rng.uniform(0, 80, n_hot)                       # metres per epoch
        start = rng.integers(0, epochs // 2, n_hot) * (rng.random(n_hot) < 0.5)
        length = np.where(rng.random(n_hot) < 0.5, epochs, rng.integers(3, epochs + 1, n_hot))
    else:
        epochs, n_hot = 12, int(n_uav * rng.uniform(2, 3))
        hot = np.where(rng.random(n_users) < 0.8, rng.integers(0, n_hot, n_users), -1)
        speed = np.zeros(n_hot)
        start = rng.integers(0, epochs - 2, n_hot)
        length = rng.integers(2, 4, n_hot)
    centre = rng.uniform(0.15 * side, 0.85 * side, size=(n_hot, 2))
    angle = rng.uniform(0, 2 * np.pi, n_hot)
    drift = speed[:, None] * np.c_[np.cos(angle), np.sin(angle)]
    spread = rng.uniform(0.04, 0.10, n_hot) * side
    offset = rng.standard_normal((n_users, 2))
    walk = rng.uniform(0, side, size=(n_users, 2))              # background users start here
    tracks, activity = [], []
    for e in range(epochs):
        def place(idx):
            c = np.clip(centre[hot[idx]] + e * drift[hot[idx]], 0.05 * side, 0.95 * side)
            return np.where(hot[idx, None] >= 0, c + spread[hot[idx], None] * offset[idx],
                            walk[idx])
        if e:
            walk = walk + rng.normal(0, 20.0, size=(n_users, 2))
            offset = offset + rng.normal(0, 0.05, size=(n_users, 2))
        xy = np.round(np.clip(place(np.arange(n_users)), 0, side), 1)
        for _ in range(400):                     # nobody stands in a building or on water
            bad = np.flatnonzero(~sc.standable(xy))
            if not len(bad):
                break
            offset[bad] = rng.standard_normal((len(bad), 2))
            walk[bad] = np.clip(walk[bad] + rng.normal(0, 40.0, size=(len(bad), 2)), 0, side)
            xy[bad] = np.round(np.clip(place(bad), 0, side), 1)
        else:
            raise ValueError("could not place every user outdoors")
        tracks.append(xy.tolist())
        on = (hot < 0) | ((start[hot] <= e) & (e < start[hot] + length[hot]))
        if not on.any():
            on[rng.integers(n_users)] = True
        activity.append(on.astype(int).tolist())
    return ub.Instance(id=f"moving-{kind}-{seed}-k{n_uav}-n{n_users}", task="place", side_m=side,
                       alt_min_m=50.0, alt_max_m=300.0, n_uav=n_uav, seed=seed, scene=scene,
                       users=[[x, y, float(d)] for (x, y), d in zip(tracks[0], demand)],
                       tracks=tracks, activity=activity, max_move_m=max_move_m)


def epoch_instance(inst: ub.Instance, e: int) -> ub.Instance:
    """One epoch as an instance of its own: the users active in it, where they are in it."""
    users = [[x, y, d] for (x, y), d in zip(inst.xy(e).tolist(), inst.demands(e).tolist())]
    return dataclasses.replace(inst, users=users, tracks=None, activity=None, max_move_m=None)


def plans(inst: ub.Instance, limits, calls: int) -> list:
    """Scores of the four kinds of plan, and the plans, for each speed limit in `limits`."""
    rng = np.random.default_rng([inst.seed, 5])
    n = inst.n_epochs
    unlimited = dataclasses.replace(inst, max_move_m=None)
    each = [epoch_instance(inst, e) for e in range(n)]

    def all_epochs(u):
        return ub.score_fast(unlimited, np.repeat(u, n, axis=0))["score"]

    static = local_search(each[0], rng, budget=calls, score=all_epochs)   # each[0]: rules, shape
    free = np.array([local_search(ie, rng, budget=calls, start=static)[0] for ie in each])
    rows = []
    for mm in limits:
        limited = dataclasses.replace(inst, max_move_m=mm)
        myopic = []
        for ie in each:
            prev = myopic[-1] if myopic else None

            def near(u, prev=prev, ie=ie):
                if prev is not None and (np.linalg.norm(u[0] - prev, axis=-1) > mm).any():
                    return -1.0
                return ub.score_fast(ie, u)["score"]
            myopic.append(local_search(ie, rng, budget=calls, score=near,
                                       start=(static if prev is None else prev[None]))[0])
        planner = local_search(limited, rng, budget=3 * calls, start=np.repeat(static, n, axis=0))
        out = {"free": free, "myopic": np.array(myopic), "planner": planner,
               "static": np.repeat(static, n, axis=0)}
        for name in ("myopic", "planner", "static"):
            ub.check(limited, out[name])
        rows.append((mm, {k: (ub.score_fast(unlimited, u)["score"], u) for k, u in out.items()}))
    return rows


def run(job):
    kind, seed, k, n, limits, channel, calls = job
    inst = moving_instance(kind, seed, k, n, limits[0])
    if channel == "statistical":
        inst = dataclasses.replace(inst, scene=None)
    return [dict(inst=inst.id, kind=kind, channel=channel, mm=mm, calls=calls,
                 **{name: round(s, 4) for name, (s, _) in res.items()},
                 plans={name: np.round(u, 1).tolist() for name, (_, u) in res.items()})
            for mm, res in plans(inst, limits, calls)]


if __name__ == "__main__":
    calls = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    done = set()
    if OUT.exists():
        done = {(r["inst"], r["channel"], r["mm"], r["calls"]) for r in map(json.loads, OUT.open())}
    jobs = [(kind, seed, k, n, mms, channel, calls)
            for kind, seed, k, n, mms in CASES for channel in ("scene", "statistical")
            if (f"moving-{kind}-{seed}-k{k}-n{n}", channel, mms[0], calls) not in done]
    with Pool(3) as pool, OUT.open("a") as f:
        for rows in pool.imap_unordered(run, jobs):
            for row in rows:
                f.write(json.dumps(row) + "\n")
                print({k: v for k, v in row.items() if k != "plans"}, flush=True)
            f.flush()
