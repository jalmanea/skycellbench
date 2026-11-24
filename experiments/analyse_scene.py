"""Summarise the two scene checks.

  python experiments/analyse_scene.py gap <set> [calls]   results/scene-gap-<set>.jsonl
  python experiments/analyse_scene.py moving [calls]      results/moving-scene.jsonl
  python experiments/analyse_scene.py limit [calls]       results/moving-limit.jsonl
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent / "results"


def gap(name, calls=None):
    rows = [json.loads(l) for l in (root / f"scene-gap-{name}.jsonl").open()]
    calls = calls or max(r["calls"] for r in rows)
    by = defaultdict(dict)
    for r in rows:
        if r["calls"] == calls:
            by[(r["task"], r["instance"])][r["channel"]] = r
    print(f"k-means against a local search with {calls} calls; closed-form scores")
    for task in sorted({t for t, _ in by}):
        g = np.array([[d[c][k] for c in ("scene", "statistical") for k in ("kmeans", "reference")]
                      for (t, _), d in sorted(by.items()) if t == task and len(d) == 2])
        gs, go = g[:, 1] - g[:, 0], g[:, 3] - g[:, 2]
        print(f"\n{task}: {len(g)} instances")
        print(f"  scene        k-means {g[:, 0].mean():.3f}  reference {g[:, 1].mean():.3f}  "
              f"gap {gs.mean():.3f} (min {gs.min():.3f}, max {gs.max():.3f})")
        print(f"  old channel  k-means {g[:, 2].mean():.3f}  reference {g[:, 3].mean():.3f}  "
              f"gap {go.mean():.3f} (min {go.min():.3f}, max {go.max():.3f})")
        print(f"  gap on the scene minus gap on the old channel: mean {np.mean(gs - go):+.3f}; "
              f"larger on the scene in {int((gs > go).sum())} of {len(g)} instances")


def moving(calls=None):
    rows = [json.loads(l) for l in (root / "moving-scene.jsonl").open()]
    calls = calls or max(r["calls"] for r in rows)
    rows = [r for r in rows if r["calls"] == calls]
    print(f"local searches with {calls} calls; closed-form scores; free = no speed limit")
    for kind in sorted({r["kind"] for r in rows}):
        for channel in ("scene", "statistical"):
            rs = sorted((r for r in rows if r["kind"] == kind and r["channel"] == channel),
                        key=lambda r: (r["inst"], -r["mm"]))
            if not rs:
                continue
            print(f"\n{kind}, {'scene' if channel == 'scene' else 'old channel, same users'}")
            print(f"  {'instance':<26}{'mm':>5}{'static':>8}{'myopic':>8}{'planner':>9}{'free':>7}"
                  f"{'free-static':>13}{'planner-static':>16}")
            for r in rs:
                print(f"  {r['inst']:<26}{r['mm']:>5.0f}{r['static']:>8.3f}{r['myopic']:>8.3f}"
                      f"{r['planner']:>9.3f}{r['free']:>7.3f}{r['free'] - r['static']:>13.3f}"
                      f"{r['planner'] - r['static']:>16.3f}")
            one = {r["inst"]: r["free"] - r["static"] for r in rs}      # same for every limit
            d = np.array(list(one.values()))
            p = np.array([r["planner"] - r["static"] for r in rs])
            print(f"  free minus static: mean {d.mean():.3f}, min {d.min():.3f}, max {d.max():.3f}; "
                  f"above 0.05 on {int((d > 0.05).sum())} of {len(d)} instances. "
                  f"planner minus static: mean {p.mean():.3f}, max {p.max():.3f}")


def limit(calls=None):
    rows = [json.loads(l) for l in (root / "moving-limit.jsonl").open()]
    calls = calls or max(r["calls"] for r in rows)
    rows = [r for r in rows if r["calls"] == calls]
    limits = sorted({r["limit"] for r in rows if r["limit"] is not None}) + [None]
    head = "".join(f"{'none' if l is None else format(l, '.0f'):>7}" for l in limits[1:])
    print(f"one search under every speed limit ({calls} calls per round); closed-form scores\n"
          f"columns: score with fixed UAVs, then the gain over it at each limit, metres per epoch")
    for kind in sorted({r["kind"] for r in rows}):
        for channel in ("scene", "statistical"):
            by = defaultdict(dict)
            for r in rows:
                if r["kind"] == kind and r["channel"] == channel:
                    by[r["inst"]][r["limit"]] = r
            by = {i: d for i, d in by.items() if len(d) == len(limits)}
            if not by:
                continue
            print(f"\n{kind}, {'scene' if channel == 'scene' else 'old channel, same users'}")
            print(f"  {'instance':<26}{'fixed':>7}{head}   free plan step, median / 90% (m)")
            g = []
            for inst, d in sorted(by.items()):
                g.append([d[l]["score"] - d[0.0]["score"] for l in limits[1:]])
                print(f"  {inst:<26}{d[0.0]['score']:>7.3f}" + "".join(f"{v:>7.3f}" for v in g[-1])
                      + f"   {d[None]['free_step_median']:.0f} / {d[None]['free_step_p90']:.0f}")
            g = np.array(g)
            print(f"  {'mean':<26}{np.mean([d[0.0]['score'] for d in by.values()]):>7.3f}"
                  + "".join(f"{v:>7.3f}" for v in g.mean(0)))
            print(f"  {'share of the free gain':<33}"
                  + "".join(f"{v:>7.0%}" for v in g.mean(0) / g.mean(0)[-1]))
            print(f"  {'instances with gain > 0.05':<33}"
                  + "".join(f"{int(v):>7d}" for v in (g > 0.05).sum(0)))
            fixed = np.array([d[0.0]["fixed_by_round"] for d in by.values()])
            old = np.array([[d[0.0]["old_static"], d[0.0]["old_free"]] for d in by.values()])
            print(f"  fixed UAVs, mean score after rounds 2, 4, 6: {fixed[:, 1].mean():.3f}, "
                  f"{fixed[:, 3].mean():.3f}, {fixed[:, 5].mean():.3f}; in moving_scene.py: "
                  f"static {old[:, 0].mean():.3f}, free {old[:, 1].mean():.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "limit":
        limit(int(sys.argv[2]) if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "gap":
        gap(sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else None)
    else:
        moving(int(sys.argv[2]) if len(sys.argv) > 2 else None)
