"""Summarise results/agreement.jsonl: how well the closed-form scorer predicts the ns-3 scorer."""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

rows = [json.loads(l) for l in (Path(__file__).parent / "results" / "agreement.jsonl").open()]
by = defaultdict(list)
for r in rows:
    by[r["instance"]].append(r)

print(f"{'instance':<18}{'n':>4}{'spearman':>10}{'kendall':>9}{'mean |diff|':>13}{'max |diff|':>12}"
      f"{'bias':>8}{'top-5 overlap':>15}")
for name, rs in by.items():
    f = np.array([r["fast"] for r in rs])
    n = np.array([r["ns3"] for r in rs])
    top = len(set(np.argsort(-f)[:5]) & set(np.argsort(-n)[:5]))
    print(f"{name:<18}{len(rs):>4}{spearmanr(f, n)[0]:>10.3f}{kendalltau(f, n)[0]:>9.3f}"
          f"{np.abs(f - n).mean():>13.3f}{np.abs(f - n).max():>12.3f}{(f - n).mean():>8.3f}"
          f"{top:>13}/5")

print("\nmean score by kind of placement (closed-form / ns-3):")
kinds = defaultdict(list)
for r in rows:
    kinds[r["kind"]].append((r["fast"], r["ns3"]))
for k, v in kinds.items():
    a = np.array(v)
    print(f"  {k:<18}{len(a):>4}  {a[:, 0].mean():.3f} / {a[:, 1].mean():.3f}")
print(f"\nns-3 wall time per score: median {np.median([r['wall_s'] for r in rows]):.1f} s "
      f"(runs were {4} at a time)")
