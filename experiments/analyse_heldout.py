"""Tables of a held-out run, as PROTOCOL.md section 6 says.

  python experiments/analyse_heldout.py <set>        reads results/heldout/<set>.jsonl
Per task and track: mean ns-3 score over the instances with a 95% bootstrap interval (10,000
resamples over instances), the closed-form score beside it, invalid answers, scorer calls,
sum throughput, 5th-percentile user and controller wall time. A controller run with 3 seeds
scores the mean of its 3 answers on each instance. Then paired differences between controllers,
by instance, with their intervals and the number of instances won. The reference (the
200,000-call search, closed-form) is shown as a level when results/headroom-<set>.jsonl has it.
Writes the same text to results/heldout/<set>.txt.
"""
import json
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np

RES = Path(__file__).resolve().parent / "results"
NAMES = {"centre": "Do-nothing", "random": "Random", "kmeans": "k-means", "search30": "Search, 30",
         "rl": "RL", "rl_scene": "RL, scene-trained", "search": "Search, 4,000"}
ORDER = list(NAMES)


def boot(x, rng, n=10_000):
    x = np.asarray(x, float)
    m = rng.choice(x, size=(n, len(x))).mean(axis=1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def main():
    name = sys.argv[1]
    rows = [json.loads(l) for l in (RES / "heldout" / f"{name}.jsonl").open()]
    ref = {}
    if (RES / f"headroom-{name}.jsonl").exists():
        ref = {r["instance"]: r["fast"] for r in map(json.loads, (RES / f"headroom-{name}.jsonl").open())}
    commits = {(r["commit"], r["protocol_commit"], r["tree_clean"]) for r in rows}
    out = [f"{name}: {len(rows)} answers; commit, protocol commit, clean tree: {sorted(commits)}"]
    # per (task, track, controller, instance): the mean over seeds of each field
    cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (r["task"], r["track"], r["controller"])
        cell[key][r["instance"]].append(r)
    rng = np.random.default_rng(0)
    for task in ("place", "constrained", "blind"):
        keys = sorted([k for k in cell if k[0] == task],
                      key=lambda k: (["low", "high", "transfer"].index(k[1]), ORDER.index(k[2])))
        if not keys:
            continue
        insts = sorted(set().union(*(cell[k].keys() for k in keys)))
        out.append(f"\n{task}: {len(insts)} instances")
        out.append(f"  {'track':<9}{'controller':<19}{'ns-3 [95% interval]':<24}{'closed-form':>12}"
                   f"{'invalid':>9}{'calls':>7}{'sum thr':>9}{'p5 user':>9}{'wall s':>8}")
        per = {}
        for k in keys:
            by = cell[k]
            mean = lambda f: np.array([np.mean([a.get(f, 0.0) or 0.0 for a in by[i]]) for i in insts])
            per[k] = dict(zip(insts, mean("ns3")))
            m, lo, hi = boot(mean("ns3"), rng)
            inv = sum(not a["valid"] for i in insts for a in by[i])
            n_ans = sum(len(by[i]) for i in insts)
            out.append(f"  {k[1]:<9}{NAMES[k[2]]:<19}{f'{m:.3f} [{lo:.3f}, {hi:.3f}]':<24}"
                       f"{mean('fast').mean():>12.3f}{f'{inv}/{n_ans}':>9}"
                       f"{mean('calls').mean():>7.0f}{mean('ns3_sum_thr_mbps').mean():>9.1f}"
                       f"{mean('ns3_p5').mean():>9.3f}{mean('controller_wall_s').mean():>8.1f}")
        r = [ref[i] for i in insts if i in ref]
        if r:
            m, lo, hi = boot(r, rng)
            out.append(f"  {'level':<9}{'Reference':<19}{f'({m:.3f}) [{lo:.3f}, {hi:.3f}]':<24}"
                       f"   closed-form, {len(r)} of {len(insts)} instances")
        out.append(f"\n  paired by instance, ns-3: row minus column, mean [95% interval], "
                   f"instances won")
        for a, b in combinations(keys, 2):
            if a[1] != b[1] and "transfer" not in (a[1], b[1]) and \
                    not (a[2] == "search" and b[2] == "search30") and \
                    not (b[2] == "search" and a[2] == "search30"):
                continue                        # across tracks: only search against search30
            d = np.array([per[a][i] - per[b][i] for i in insts])
            m, lo, hi = boot(d, rng)
            sure = "" if lo <= 0 <= hi else "  *"
            out.append(f"  {NAMES[a[2]] + (' (transfer)' if a[1] == 'transfer' else ''):<30} minus "
                       f"{NAMES[b[2]] + (' (transfer)' if b[1] == 'transfer' else ''):<30}"
                       f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]  {int((d > 0).sum())}/{len(d)}{sure}")
    out.append("\n* the interval does not include 0")
    text = "\n".join(out)
    (RES / "heldout" / f"{name}.txt").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
