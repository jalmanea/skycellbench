"""Summarise results/compare-<set>.jsonl: mean scores per task and controller on both scorers,
and how well the closed-form scorer predicts ns-3.   python experiments/analyse_compare.py public
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

rows = [json.loads(l) for l in
        (Path(__file__).parent / "results" / f"compare-{sys.argv[1]}.jsonl").open()]
rows = [r for r in rows if "ns3" in r]
order = ["centre", "random", "kmeans", "rl", "rl_scene", "search30", "search"]
for task in sorted({r["task"] for r in rows}):
    rs = [r for r in rows if r["task"] == task]
    f, n = np.array([r["fast"] for r in rs]), np.array([r["ns3"] for r in rs])
    print(f"\n{task}: {len(rs)} answers, Spearman {spearmanr(f, n)[0]:.3f}, "
          f"closed-form minus ns-3: mean {np.mean(f - n):+.3f}, max {np.max(f - n):+.3f}, "
          f"min {np.min(f - n):+.3f}")
    by = defaultdict(dict)
    for r in rs:
        by[r["instance"]][r["controller"]] = r
    for c in order:
        v = [(i[c]["fast"], i[c]["ns3"]) for i in by.values() if c in i]
        if v:
            a = np.array(v)
            print(f"  {c:<9} closed-form {a[:, 0].mean():.3f}   ns-3 {a[:, 1].mean():.3f}")
    # does ns-3 keep the order of each pair of controllers on each instance?
    same = total = 0
    rho = []
    for i in by.values():
        cs = [c for c in order if c in i]
        rho.append(spearmanr([i[c]["fast"] for c in cs], [i[c]["ns3"] for c in cs])[0])
        for x in range(len(cs)):
            for y in range(x + 1, len(cs)):
                df, dn = (i[cs[x]][k] - i[cs[y]][k] for k in ("fast", "ns3"))
                if abs(df) > 1e-9:
                    total += 1
                    same += (df > 0) == (dn > 0)
    print(f"  order of controller pairs kept by ns-3: {same}/{total}; "
          f"Spearman within an instance, mean {np.mean(rho):.2f}, min {np.min(rho):.2f}")
