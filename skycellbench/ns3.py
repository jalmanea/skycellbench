"""ns-3 LTE scorer: runs the packet-level simulation for one placement (about 20 s)."""
from __future__ import annotations

import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .fast import UE_HEIGHT_M, epoch_mean, path_loss_db
from .instance import Instance, check

BINARY = Path(os.environ.get("SKYCELLBENCH_NS3_BIN",
                             Path(__file__).resolve().parent.parent / "ns3" / "bin" / "uav-score"))


def write_scenario(inst: Instance, u, xy, demand, path) -> None:
    """Flat text the C++ program reads: radio settings, UAV positions, users with demands, and
    the path loss table of fast.path_loss_db, one row per UAV. ns-3 computes no loss itself."""
    r = inst.radio
    loss = path_loss_db(inst, u, xy)
    sc = inst.get_scene()
    uz = (sc.ground(xy) if sc is not None else 0.0) + np.full(len(xy), UE_HEIGHT_M)
    lines = [f"radio2 {r.rb} {r.tx_power_dbm} {r.noise_figure_db}", f"uavs {len(u)}"]
    lines += [f"{x!r} {y!r} {z!r}" for x, y, z in inst.absolute(u).tolist()]
    lines.append(f"users {len(xy)}")
    lines += [f"{x!r} {y!r} {z!r} {d!r}"
              for (x, y), z, d in zip(xy.tolist(), uz.tolist(), demand.tolist())]
    lines.append("loss")
    lines += [" ".join(repr(v) for v in row) for row in loss.tolist()]
    Path(path).write_text("\n".join(lines) + "\n")


def _run(inst: Instance, u, xy, demand, sim_time: float, run: int) -> dict:
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        path = f.name
    try:
        write_scenario(inst, u, xy, demand, path)
        res = subprocess.run([str(BINARY), f"--scenario={path}", f"--simTime={sim_time}",
                              f"--run={run}"], capture_output=True, text=True)
    finally:
        os.unlink(path)
    if res.returncode != 0:
        raise RuntimeError(f"ns-3 scorer failed: {res.stderr[-2000:]}")
    return json.loads(res.stdout.strip().splitlines()[-1])


def score_ns3(inst: Instance, uavs, sim_time: float = 2.0, run: int = 1, detail: bool = False,
              parallel: int = 1):
    """Score a placement with ns-3; with several epochs, one simulation per epoch, then the mean."""
    if not BINARY.exists():
        raise FileNotFoundError(f"{BINARY} not built; see ns3/build.sh")
    u = check(inst, uavs)
    with ThreadPoolExecutor(parallel) as ex:
        outs = list(ex.map(lambda e: _run(inst, u[e], inst.xy(e), inst.demands(e), sim_time, run), range(len(u))))
    if len(outs) == 1:
        return outs[0] if detail else {"score": outs[0]["score"]}
    out = {"score": epoch_mean(inst, [o["score"] for o in outs])}
    if detail:
        out["epochs"] = outs
    return out
