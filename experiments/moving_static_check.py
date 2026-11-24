"""How much of "free minus static" in moving_scene.py is only more search?

  PYTHONPATH=. python experiments/moving_static_check.py
The free plan gets `calls` scorer calls per epoch and the static plan `calls` in all. This
repeats the static search with four times the calls on four scene instances and prints the
difference to the free plan before and after. Reads results/moving-scene.jsonl.
"""
import sys, dataclasses, json, numpy as np
from multiprocessing import Pool
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import moving_scene as ms, skycellbench as ub
from skycellbench.baselines import local_search

def run(job):
    kind, seed, k, n, mm = job
    inst = ms.moving_instance(kind, seed, k, n, mm)
    un = dataclasses.replace(inst, max_move_m=None); one = ms.epoch_instance(inst, 0)
    f = lambda u: ub.score_fast(un, np.repeat(u, inst.n_epochs, axis=0))["score"]
    st = local_search(one, np.random.default_rng([inst.seed, 5]), budget=12000, score=f)
    return inst.id, f(st)

if __name__ == "__main__":
    rows = {(r["inst"], r["channel"], r["mm"]): r for r in map(json.loads, ms.OUT.open())}
    jobs = [("drift", 3, 8, 250, 100.0), ("drift", 4, 5, 150, 100.0), ("events3", 1, 3, 150, 150.0), ("events3", 4, 5, 150, 150.0)]
    with Pool(4) as pool:
        for name, s in pool.imap_unordered(run, jobs):
            r = next(v for (i, c, m), v in rows.items() if i == name and c == "scene")
            print(f"{name}: static 3000 calls {r['static']:.3f}, 12000 calls {s:.3f}, free {r['free']:.3f}; free minus static {r['free'] - r['static']:.3f} -> {r['free'] - s:.3f}", flush=True)
