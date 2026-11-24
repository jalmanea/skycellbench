"""Write the examples the viewer opens by default (viewer/examples).

  PYTHONPATH=. python experiments/viewer_examples.py
For a few public instances: the k-means answer and the answer of a 4,000-call local search,
each saved with what every user received (closed-form scorer), as `python -m skycellbench view`
does. Only public instances are used. One more example has several epochs: an instance of the
moving-users test (moving_scene.py) with two of the plans found for it.
"""
import json
import shutil
import sys
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench import controllers
from skycellbench.__main__ import view_result

sys.path.insert(0, str(Path(__file__).resolve().parent))
from moving_scene import moving_instance  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "viewer" / "examples"
PICK = ["public-cincinnati-downtown/place-cincinnati-downtown-100002-k8-n250",
        "public-cincinnati-downtown/place-cincinnati-downtown-100000-k3-n150",
        "public-cincinnati-downtown/constrained-cincinnati-downtown-100001-k5-n200",
        "public/place-100001-k5-n200"]

OUT.mkdir(parents=True, exist_ok=True)
index = []
for name in PICK:
    src = ROOT / "instances" / f"{name}.json"
    inst = ub.Instance.load(src)
    shutil.copy(src, OUT / src.name)
    where = "Cincinnati" if inst.scene else "flat ground"
    for cname, title in (("kmeans", "k-means"), ("search", "search, 4,000 calls")):
        u = controllers.ALL[cname](inst, None, np.random.default_rng([inst.seed, len(cname)]))
        place = ub.Placement(inst.id, np.round(u, 2).tolist())
        place.result = view_result(inst, place.uavs)
        file = f"{inst.id}.{cname}.json"
        place.save(OUT / file)
        index.append({"label": f"{where}, {inst.task}, {inst.n_uav} UAVs: {title} "
                               f"({place.result['score']:.2f})",
                      "instance": src.name, "answer": file})
        print(index[-1]["label"])

rows = [json.loads(l) for l in (ROOT / "experiments" / "results" / "moving-scene.jsonl").open()]
row = next(r for r in rows if r["inst"] == "moving-drift-3-k8-n250" and r["channel"] == "scene"
           and r["mm"] == 100.0)
inst = moving_instance("drift", 3, 8, 250, 100.0)
inst.save(OUT / f"{inst.id}.json")
for plan, title in (("static", "UAVs stay put"), ("myopic", "UAVs follow, 100 m per epoch")):
    place = ub.Placement(inst.id, row["plans"][plan])
    place.result = view_result(inst, place.uavs)
    file = f"{inst.id}.{plan}.json"
    place.save(OUT / file)
    index.append({"label": f"Cincinnati, moving users, 8 UAVs, {inst.n_epochs} epochs: {title} "
                           f"({place.result['score']:.2f})",
                  "instance": f"{inst.id}.json", "answer": file})
    print(index[-1]["label"])
(OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
