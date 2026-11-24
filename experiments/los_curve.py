"""How much line of sight does the scene give, against what the old channel assumed?

  PYTHONPATH=. python experiments/los_curve.py [set]        default public-cincinnati-downtown
Part 1, the whole scene. Random outdoor points and random UAV positions (50 to 300 m above the
ground, not inside a building): the share of links with line of sight per band of elevation
angle, next to the line-of-sight probability of the old channel at the same angles, and the two
parameters of that probability curve fitted to the scene.
Part 2, the answers of the gap check (results/scene-gap-<set>.jsonl). For the k-means answer and
the reference answer of every instance, on the scene and on the old channel: how far and how steep
the serving links are, how much excess loss they carry, and how much weaker the strongest other
UAV is than the serving one. Each answer is the one found on that channel.
"""
import dataclasses
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import skycellbench as ub
from skycellbench.fast import UE_HEIGHT_M, path_loss_db

root = Path(__file__).resolve().parent.parent
name = sys.argv[1] if len(sys.argv) > 1 else "public-cincinnati-downtown"
SCENE = "cincinnati-downtown"
BANDS = (0, 3, 6, 10, 15, 20, 30, 45, 60, 90)
radio = ub.instance.Radio()


def model(theta, a=radio.a, b=radio.b):
    return 1.0 / (1.0 + a * np.exp(-b * (theta - a)))


def links(sc, uavs, xy):
    """Line of sight, elevation angle (degrees) and ground distance of every UAV-user link."""
    px = np.column_stack([xy, sc.ground(xy) + UE_HEIGHT_M])
    pu = np.array(uavs, float)
    pu[:, 2] += sc.ground(pu[:, :2])
    los = np.array([sc.los(p, px) for p in pu])
    ground = np.hypot(pu[:, None, 0] - px[None, :, 0], pu[:, None, 1] - px[None, :, 1])
    theta = np.degrees(np.arctan2(pu[:, None, 2] - px[None, :, 2], np.maximum(ground, 1e-9)))
    return los, theta, ground


def table(title, los, theta):
    print(f"\n{title}: {los.size} links, line of sight on {los.mean():.1%}; "
          f"the old channel expects {model(theta).mean():.1%} at the same angles")
    print(f"  {'elevation':>12}{'links':>9}{'scene':>8}{'old channel':>13}")
    for lo, hi in zip(BANDS, BANDS[1:]):
        m = (theta >= lo) & (theta < hi)
        if m.sum() >= 50:
            print(f"  {f'{lo}-{hi} deg':>12}{int(m.sum()):>9}{los[m].mean():>8.1%}"
                  f"{model(theta[m]).mean():>13.1%}")
    m = theta > 0
    t, y = theta[m], los[m]

    def nll(p):
        q = np.clip(model(t, np.exp(p[0]), np.exp(p[1])), 1e-9, 1 - 1e-9)
        return -(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()
    fit = minimize(nll, np.log([radio.a, radio.b]), method="Nelder-Mead")
    a, b = np.exp(fit.x)
    half = lambda a, b: a + np.log(a) / b               # angle with line of sight half the time
    print(f"  fitted curve: a = {a:.2f}, b = {b:.3f} (old channel: a = {radio.a}, b = {radio.b}); "
          f"line of sight half the time at {half(a, b):.0f} deg, old channel "
          f"{half(radio.a, radio.b):.0f} deg")


def whole_scene():
    sc = ub.get_scene(SCENE)
    rng = np.random.default_rng(0)
    side = sc.size[0]
    xy = rng.uniform(0, side, size=(8000, 2))
    xy = xy[sc.standable(xy)][:2000]
    u = np.column_stack([rng.uniform(0, side, size=(900, 2)), rng.uniform(50, 300, 900)])
    u = u[u[:, 2] > sc.building_height(u[:, :2])][:600]
    los, theta, _ = links(sc, u, xy)
    table("whole scene, random outdoor points", los, theta)
    xs = np.vstack([ub.Instance.load(p).xy() for p in sorted((root / "instances" / name).glob("*.json"))])
    los, theta, _ = links(sc, u, xs[rng.choice(len(xs), 2000, replace=False)])
    table(f"users of {name}", los, theta)
    print("\n  by altitude above ground, users of the set:")
    for lo, hi in ((50, 100), (100, 200), (200, 300)):
        m = (u[:, 2] >= lo) & (u[:, 2] < hi)
        print(f"  {lo}-{hi} m: scene {los[m].mean():.1%}, old channel {model(theta[m]).mean():.1%}")


def answers():
    rows = [json.loads(l) for l in (root / "experiments" / "results" / f"scene-gap-{name}.jsonl").open()]
    calls = max(r["calls"] for r in rows)
    insts = {p.stem: ub.Instance.load(p) for p in (root / "instances" / name).glob("*.json")}
    out = defaultdict(list)
    for r in rows:
        if r["calls"] != calls:
            continue
        inst = insts[r["instance"]]
        flat = dataclasses.replace(inst, scene=None)
        xy = inst.xy()
        for kind in ("kmeans", "reference"):
            u = np.array(r[f"{kind}_uavs"], float)[0]
            use = inst if r["channel"] == "scene" else flat
            loss = path_loss_db(use, u, xy)
            bare = path_loss_db(dataclasses.replace(use, radio=dataclasses.replace(
                use.radio, eta_los_db=0.0, eta_nlos_db=0.0)), u, xy)
            n = np.arange(len(xy))
            s = loss.argmin(0)
            if r["channel"] == "scene":
                los, theta, ground = links(inst.get_scene(), u, xy)
                sight_all, sight_serving = los.mean(), los[s, n].mean()
            else:
                ground = np.hypot(u[:, None, 0] - xy[None, :, 0], u[:, None, 1] - xy[None, :, 1])
                theta = np.degrees(np.arctan2(u[:, 2:3] - UE_HEIGHT_M, np.maximum(ground, 1e-9)))
                sight_all, sight_serving = model(theta).mean(), model(theta)[s, n].mean()
            two = np.sort(loss, axis=0)
            out[(r["task"], kind, r["channel"])].append(
                [r[kind], ground[s, n].mean(), theta[s, n].mean(), sight_all, sight_serving,
                 (loss - bare)[s, n].mean(), (two[1] - two[0]).mean(), u[:, 2].mean()])
    print(f"\n\nanswers of the gap check ({calls} calls), mean over the instances of each task")
    print("sight = share of links with line of sight (scene) or mean line-of-sight probability "
          "(old channel);\nexcess = excess loss of the serving link; margin = loss of the "
          "second-best UAV minus loss of the serving one")
    print(f"\n  {'task':<12}{'answer':<10}{'channel':<13}{'score':>6}{'alt m':>7}{'dist m':>8}"
          f"{'angle':>7}{'sight all':>11}{'serving':>9}{'excess dB':>11}{'margin dB':>11}")
    for (task, kind, channel), v in sorted(out.items()):
        m = np.mean(v, axis=0)
        print(f"  {task:<12}{kind:<10}{'scene' if channel == 'scene' else 'old channel':<13}"
              f"{m[0]:>6.3f}{m[7]:>7.0f}{m[1]:>8.0f}{m[2]:>7.1f}{m[3]:>11.1%}{m[4]:>9.1%}"
              f"{m[5]:>11.1f}{m[6]:>11.1f}")


if __name__ == "__main__":
    whole_scene()
    answers()
