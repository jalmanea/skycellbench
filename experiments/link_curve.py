"""Measure the ns-3 LTE link curve: downlink rate of one user alone in a cell against SINR.

ns-3 picks the modulation and coding from mutual-information error curves, which have no closed
form, so the closed-form scorer reads this measured table instead. One user, one UAV, no
interference; the SINR is swept by changing the transmit power. Writes skycellbench/link_curve.json.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import skycellbench as ub
from skycellbench.fast import RB_HZ, path_loss_db

UAV = [[1000.0, 1000.0, 100.0]]


def one(sinr_db: float, rb: int = 100) -> float:
    inst = ub.Instance(id="link", side_m=2000, alt_min_m=50, alt_max_m=300, n_uav=1,
                       users=[[1000.0, 1000.0, 150.0]])
    inst.radio.rb = rb
    loss = float(path_loss_db(inst, np.array(UAV))[0, 0])
    noise_dbm = -174.0 + inst.radio.noise_figure_db + 10 * np.log10(RB_HZ * rb)
    inst.radio.tx_power_dbm = sinr_db + loss + noise_dbm
    out = ub.score_ns3(inst, UAV, sim_time=2.0, detail=True)
    return out["thr_mbps"][0] * 1e6 / (RB_HZ * rb)          # bit/s/Hz


if __name__ == "__main__":
    grid = np.round(np.arange(-7.0, 24.01, 0.25), 2)
    with ThreadPoolExecutor(6) as ex:
        se = list(ex.map(one, grid))
    out = {"note": "ns-3.48 LTE, 100 RB, one user, UDP downlink, measured by link_curve.py",
           "sinr_db": grid.tolist(), "bit_per_s_per_hz": [round(v, 4) for v in se]}
    Path(ub.__file__).with_name("link_curve.json").write_text(json.dumps(out) + "\n")
    for s, v in zip(grid[::4], se[::4]):
        print(f"{s:6.2f} dB  {v:5.2f} bit/s/Hz")
