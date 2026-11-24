"""Command line: generate an instance, score a placement, or run a simple baseline.

  python -m skycellbench generate --seed 7 --task constrained -o inst.json
  python -m skycellbench baseline inst.json --method kmeans -o place.json
  python -m skycellbench score inst.json place.json --tier fast      (or --tier ns3)
  python -m skycellbench view inst.json place.json -o scored.json    the answer with what each
                                                                     user received, for viewer/
"""
import argparse
import json

import numpy as np

from . import TASKS, Instance, Placement, baselines, generate, score_fast, score_ns3


def view_result(inst, uavs, tier="fast") -> dict:
    """Per epoch and per active user: throughput, serving UAV and, on a scene, line of sight.

    Line of sight and the serving UAV come from the loss table, which both scorers share; the
    throughput comes from the scorer named by `tier`.
    """
    fast = score_fast(inst, uavs, detail=True)
    slow = score_ns3(inst, uavs, detail=True) if tier == "ns3" else fast
    pick = ("score", "thr_mbps", "serving", "los", "uav_users", "uav_demand_mbps",
            "uav_delivered_mbps")
    epochs = []
    for f, s in zip(fast.get("epochs", [fast]), slow.get("epochs", [slow])):
        e = {**f, "score": s["score"], "thr_mbps": s["thr_mbps"]}
        if tier == "ns3":
            e["uav_delivered_mbps"] = np.bincount(f["serving"], s["thr_mbps"],
                                                  minlength=inst.n_uav).tolist()
        epochs.append({k: e[k] for k in pick if k in e})
    return {"tier": tier, "score": slow["score"], "epochs": epochs}


def main():
    ap = argparse.ArgumentParser(prog="skycellbench")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--seed", type=int, required=True)
    g.add_argument("--task", choices=TASKS, default="place")
    g.add_argument("--public", action="store_true",
                   help="write the controller's copy (blind task: without the users)")
    g.add_argument("--n-uav", type=int, default=5)
    g.add_argument("--n-users", type=int, default=200)
    g.add_argument("--scene", help="id of a scene in scenes/, e.g. cincinnati-downtown")
    g.add_argument("-o", "--out", required=True)
    s = sub.add_parser("score")
    s.add_argument("instance")
    s.add_argument("placement")
    s.add_argument("--tier", choices=["fast", "ns3"], default="fast")
    s.add_argument("--detail", action="store_true")
    v = sub.add_parser("view")
    v.add_argument("instance")
    v.add_argument("placement")
    v.add_argument("--tier", choices=["fast", "ns3"], default="fast")
    v.add_argument("-o", "--out", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("instance")
    b.add_argument("--method", choices=["centre", "random", "kmeans", "search"], required=True)
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--budget", type=int, default=2000)
    b.add_argument("-o", "--out", required=True)
    a = ap.parse_args()

    if a.cmd == "generate":
        inst = generate(a.seed, task=a.task, n_uav=a.n_uav, n_users=a.n_users, scene=a.scene)
        (inst.public() if a.public else inst).save(a.out)
    elif a.cmd == "score":
        inst, place = Instance.load(a.instance), Placement.load(a.placement)
        if place.instance != inst.id:
            raise SystemExit(f"placement is for {place.instance}, not {inst.id}")
        fn = score_fast if a.tier == "fast" else score_ns3
        print(json.dumps(fn(inst, place.uavs, detail=a.detail)))
    elif a.cmd == "view":
        inst, place = Instance.load(a.instance), Placement.load(a.placement)
        if place.instance != inst.id:
            raise SystemExit(f"placement is for {place.instance}, not {inst.id}")
        place.result = view_result(inst, place.uavs, a.tier)
        place.save(a.out)
    else:
        inst, rng = Instance.load(a.instance), np.random.default_rng(a.seed)
        u = {"centre": lambda: baselines.centre_placement(inst),
             "random": lambda: baselines.random_placement(inst, rng),
             "kmeans": lambda: baselines.kmeans_placement(inst, rng),
             "search": lambda: baselines.local_search(inst, rng, budget=a.budget)}[a.method]()
        Placement(inst.id, np.round(u, 2).tolist()).save(a.out)


if __name__ == "__main__":
    main()
