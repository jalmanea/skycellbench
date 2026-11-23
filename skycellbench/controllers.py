"""The controllers that are compared, behind one call: controller(instance, probe, rng) -> answer.

`instance` is the public copy. `probe` is None except on the blind task, where it is the only
way to learn about the users and the controller may not exceed its budget.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import baselines as b
from .fast import score_fast

MODELS = Path(__file__).resolve().parent.parent / "experiments" / "models"


def centre(inst, probe, rng):
    return b.centre_placement(inst)


def random(inst, probe, rng):
    return b.random_placement(inst, rng)


def kmeans(inst, probe, rng):
    """k-means at 100, 150 and 200 m; keeps the best by the closed-form scorer. Not for blind."""
    if probe is not None:
        return None
    cands = [b.kmeans_placement(inst, rng, a) for a in (100, 150, 200)]
    return max(cands, key=lambda u: score_fast(inst, u)["score"])


def search(inst, probe, rng):
    """Local search: 4,000 closed-form calls, or the probes alone on the blind task."""
    if probe is not None:
        return b.probe_search(inst, probe, rng)
    return b.local_search(inst, rng, budget=4000)


def search30(inst, probe, rng):
    """Local search with 30 scorer calls on every task, the budget of the blind task."""
    if probe is not None:
        return b.probe_search(inst, probe, rng)
    return b.local_search(inst, rng, budget=30)


def rl(inst, probe, rng):
    """PPO policy trained per task on generated instances; one rollout, no search."""
    from .rl import load_policy, rl_placement
    return rl_placement(inst, load_policy(MODELS / f"rl-{inst.task}.pt"), probe)


ALL = {"centre": centre, "random": random, "kmeans": kmeans, "search": search,
       "search30": search30, "rl": rl}
