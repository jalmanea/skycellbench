"""Write the instance sets.

  instances/public/    for development: anyone may look at these and tune on them
  instances/heldout/   for the reported scores: not to be opened or scored before the
                       evaluation protocol is frozen
Ten instances per task in each set, with 3, 5 or 8 UAVs and 150 to 250 users. Training may also
draw fresh instances from the generator with any seed below 100000.
"""
from pathlib import Path

import skycellbench as ub

ROOT = Path(__file__).resolve().parent.parent / "instances"
SIZES = [(3, 150), (5, 200), (8, 250), (5, 150), (3, 200), (8, 200), (5, 250), (5, 200),
         (3, 150), (8, 250)]
for name, base in (("public", 100000), ("heldout", 200000)):
    out = ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    for task in ub.TASKS:
        for i, (k, n) in enumerate(SIZES):
            inst = ub.generate(base + i, task, n_uav=k, n_users=n)
            inst.save(out / f"{inst.id}.json")
print("written", sum(1 for _ in ROOT.rglob("*.json")), "instances")
