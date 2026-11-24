"""Simple controllers: sanity checks and classical baselines. None uses a learned model.

Every baseline returns an (epochs, K, 3) array that passes the rules of the instance.
"""
from __future__ import annotations

import numpy as np

from .fast import Probe, score_fast
from .instance import Instance, check, inside_building

ROOF_CLEARANCE_M = 5.0


def _valid(inst: Instance, u) -> bool:
    try:
        check(inst, u)
        return True
    except ValueError:
        return False


def _repeat(inst: Instance, u: np.ndarray) -> np.ndarray:
    return np.repeat(u[None], inst.n_epochs, axis=0)


def _lift(inst: Instance, u: np.ndarray) -> np.ndarray:
    """On a scene, raise every UAV that is inside a building to just above its roof."""
    hit = inside_building(inst, u)
    if not hit.any():
        return u
    roof = inst.get_scene().building_height(u[..., :2]) + ROOF_CLEARANCE_M
    u = u.copy()
    u[..., 2] = np.where(hit, np.minimum(roof, inst.alt_max_m), u[..., 2])
    return u


def centre_placement(inst: Instance) -> np.ndarray:
    """Do-nothing baseline: every UAV at one safe point, mid altitude, never moving.

    The point is the centre of the area, or just above the gateway when there is one.
    """
    mid = (inst.alt_min_m + inst.alt_max_m) / 2
    x, y = inst.gateway if inst.gateway is not None else (inst.side_m / 2, inst.side_m / 2)
    z = mid if inst.gateway is None else inst.alt_min_m
    return _lift(inst, _repeat(inst, np.tile([x, y, z], (inst.n_uav, 1))))


def random_placement(inst: Instance, rng, tries: int = 2000) -> np.ndarray:
    """A random valid placement that does not move; falls back to the do-nothing placement."""
    for _ in range(tries):
        xy = rng.uniform(0, inst.side_m, size=(inst.n_uav, 2))
        z = rng.uniform(inst.alt_min_m, inst.alt_max_m, size=(inst.n_uav, 1))
        u = _repeat(inst, np.hstack([xy, z]))
        if _valid(inst, u):
            return u
    return centre_placement(inst)


def _kmeans(xy, w, c, iters=50):
    for _ in range(iters):
        lab = ((xy[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        for k in range(len(c)):
            m = lab == k
            if m.any():
                c[k] = np.average(xy[m], axis=0, weights=w[m])
    return c


def kmeans_placement(inst: Instance, rng, altitude: float = 100.0) -> np.ndarray:
    """UAVs above the demand-weighted k-means centres of the users, at one fixed altitude.

    With several epochs, the centres are tracked from epoch to epoch and each step is cut to the speed
    limit. Constrained task: the layout is shrunk towards the gateway until it is valid.
    On a scene, a UAV that would be inside a building is raised to just above its roof.
    Needs the user positions, so it cannot be used on the blind task.
    """
    c = inst.xy(0)[rng.choice(len(inst.xy(0)), inst.n_uav, replace=False)].copy()
    plan = []
    for e in range(inst.n_epochs):
        c = _kmeans(inst.xy(e), inst.demands(e), c.copy(), iters=50 if e == 0 else 10)
        p = np.hstack([c, np.full((inst.n_uav, 1), altitude)])
        if plan and inst.max_move_m is not None:
            step = p - plan[-1]
            d = np.linalg.norm(step, axis=1, keepdims=True)
            p = plan[-1] + step * np.minimum(1.0, inst.max_move_m / np.maximum(d, 1e-9))
            c = p[:, :2].copy()
        plan.append(p)
    u = _lift(inst, np.array(plan))
    if inst.gateway is not None:
        g = np.array([*inst.gateway, inst.alt_min_m])
        for t in np.linspace(1, 0, 41):
            v = g + t * (u - g)
            v[..., 2] = np.clip(v[..., 2], inst.alt_min_m, inst.alt_max_m)
            v = _lift(inst, v)
            if _valid(inst, v):
                return v
    return u if _valid(inst, u) else centre_placement(inst)


def local_search(inst: Instance, rng, budget: int = 2000, score=None, start=None) -> np.ndarray:
    """(1+1) evolution strategy with the one-fifth success rule, restarted when it stalls.

    A move shifts one UAV, in one epoch or in all epochs at once; moves that break a rule are
    rejected. `budget` counts calls to `score`, which defaults to the closed-form scorer.
    """
    score = score or (lambda u: score_fast(inst, u)["score"])
    scale = np.array([inst.side_m, inst.side_m, inst.alt_max_m - inst.alt_min_m])
    lo = np.array([0, 0, inst.alt_min_m])
    hi = np.array([inst.side_m, inst.side_m, inst.alt_max_m])
    informed = len(inst.users) > 0
    best_u, best_s, used, first = None, -1.0, 0, True
    while used < budget:
        if first and start is not None:
            u = check(inst, start)
        elif informed and (first or rng.random() < 0.5):
            u = kmeans_placement(inst, rng, altitude=rng.uniform(inst.alt_min_m, inst.alt_max_m))
        else:
            u = random_placement(inst, rng) if not first or not inst.gateway else \
                centre_placement(inst)
        first = False
        s, step, stall = score(u), 0.1, 0
        used += 1
        while used < budget and stall < 150 and step > 1e-4:
            v = u.copy()
            k = rng.integers(inst.n_uav)
            d = step * scale * rng.standard_normal(3)
            if inst.n_epochs > 1 and rng.random() < 0.5:
                v[rng.integers(inst.n_epochs), k] += d
            else:
                v[:, k] += d
            v = np.clip(v, lo, hi)
            if not _valid(inst, v):
                step, stall = max(step * 0.9, 2e-4), stall + 1
                continue
            t = score(v)
            used += 1
            if t > s:
                u, s, step, stall = v, t, step * 1.5, 0
            else:
                step, stall = step * 0.9, stall + 1
        if s > best_s:
            best_u, best_s = u, s
    return best_u


def probe_search(inst: Instance, probe: Probe, rng) -> np.ndarray:
    """Blind task: the same local search, with the probes as its only scorer."""
    budget = probe.remaining
    return local_search(inst, rng, budget=budget, score=lambda u: probe(u)["score"])
