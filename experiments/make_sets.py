"""Write the instance sets.

  python experiments/make_sets.py                       the sets on flat ground
  python experiments/make_sets.py <scene>               the sets on a scene, e.g. cincinnati-downtown
  python experiments/make_sets.py [<scene>] --heldout   the held-out set only

  instances/public/          for development: anyone may look at these and tune on them
  instances/heldout/         for the reported scores: not to be opened or scored before the
                             evaluation protocol is frozen
  instances/public-<scene>/  the same seeds on a scene, with the same rules as the two above
  instances/heldout-<scene>/
Public sets: 10 instances per task. Held-out sets: 30 per task, the same ten sizes three times
(3, 5 or 8 UAVs, 150 to 250 users). Training may also draw fresh instances from the generator
with any seed below 90000.
"""
import sys
from pathlib import Path

import skycellbench as ub

ROOT = Path(__file__).resolve().parent.parent / "instances"
SIZES = [(3, 150), (5, 200), (8, 250), (5, 150), (3, 200), (8, 200), (5, 250), (5, 200),
         (3, 150), (8, 250)]
PER_TASK = {"public": 10, "heldout": 30}
args = [a for a in sys.argv[1:] if not a.startswith("--")]
scene = args[0] if args else None
kinds = ("heldout",) if "--heldout" in sys.argv else ("public", "heldout")
for kind in kinds:
    base = 100000 if kind == "public" else 200000
    out = ROOT / (kind if scene is None else f"{kind}-{scene}")
    out.mkdir(parents=True, exist_ok=True)
    for task in ub.TASKS:
        for i in range(PER_TASK[kind]):
            k, n = SIZES[i % len(SIZES)]
            inst = ub.generate(base + i, task, n_uav=k, n_users=n, scene=scene)
            inst.save(out / f"{inst.id}.json")
    print("written", sum(1 for _ in out.glob("*.json")), "instances to", out)
