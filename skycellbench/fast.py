"""Closed-form scorer: no simulation, a few hundred microseconds per placement.

It follows the same chain as the ns-3 LTE scorer, in closed form:
  mean air-to-ground path loss -> strongest-cell association -> SINR with interference scaled by
  how busy each neighbour is -> link curve measured from ns-3 -> proportional-fair sharing
  of each cell among its users, capped at each user's demand.
The cell loads and the SINRs depend on each other, so they are solved by fixed-point iteration.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .instance import Instance, check

UE_HEIGHT_M = 1.5
RB_HZ = 180e3
# Link curve measured from ns-3 (experiments/link_curve.py): rate of a user alone in a cell, in
# bit/s/Hz, against SINR. It is a staircase, one step per modulation and coding scheme.
_curve = json.loads(Path(__file__).with_name("link_curve.json").read_text())
CURVE_SINR_DB = np.asarray(_curve["sinr_db"])
CURVE_SE = np.maximum.accumulate(_curve["bit_per_s_per_hz"])   # smooth out HARQ noise


def spectral_efficiency(sinr_db: np.ndarray) -> np.ndarray:
    i = np.searchsorted(CURVE_SINR_DB, sinr_db + 1e-9, side="right") - 1
    return np.where(i < 0, 0.0, CURVE_SE[np.clip(i, 0, len(CURVE_SE) - 1)])


def path_loss_db(inst: Instance, uavs: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Mean air-to-ground loss, shape (K, N): free space plus the expected excess loss."""
    r = inst.radio
    d2 = uavs[:, None, :2] - xy[None, :, :]
    ground = np.hypot(d2[..., 0], d2[..., 1])
    dz = np.abs(uavs[:, 2:3] - UE_HEIGHT_M)
    dist = np.maximum(1.0, np.hypot(ground, dz))
    theta = np.degrees(np.arctan2(dz, np.maximum(ground, 1e-9)))
    p_los = 1.0 / (1.0 + r.a * np.exp(-r.b * (theta - r.a)))
    fspl = 20.0 * np.log10(4.0 * np.pi * r.freq_hz * dist / 299792458.0)
    return fspl + p_los * r.eta_los_db + (1.0 - p_los) * r.eta_nlos_db


def _share(need: np.ndarray) -> np.ndarray:
    """Split one cell's time among users who each need `need` of it (water-filling).

    Proportional fair gives backlogged users equal time; a user who needs less takes only that.
    """
    if need.sum() <= 1.0:
        return need
    order = np.sort(need)
    left = 1.0
    for j, v in enumerate(order):
        level = left / (len(order) - j)
        if v > level:
            return np.minimum(need, level)
        left -= v
    return need


def score_fast(inst: Instance, uavs, detail: bool = False):
    """Score a placement; with several epochs, the mean over all active users in all epochs."""
    u = check(inst, uavs)
    outs = [_snapshot(inst, u[e], inst.xy(e), inst.demands(e), detail) for e in range(len(u))]
    if len(outs) == 1:
        return outs[0]
    out = {"score": epoch_mean(inst, [o["score"] for o in outs])}
    if detail:
        out["epochs"] = outs
    return out


def epoch_mean(inst: Instance, scores) -> float:
    w = [inst.active(e).sum() for e in range(inst.n_epochs)]
    return float(np.average(scores, weights=w))


def _snapshot(inst: Instance, u: np.ndarray, xy: np.ndarray, demand: np.ndarray, detail: bool):
    r = inst.radio
    k, n = inst.n_uav, len(xy)
    loss = path_loss_db(inst, u, xy)
    serving = loss.argmin(axis=0)
    p_rb_mw = 10 ** (r.tx_power_dbm / 10) / r.rb
    noise_mw = 10 ** ((-174.0 + r.noise_figure_db) / 10) * RB_HZ
    rx = p_rb_mw * 10 ** (-loss / 10)                       # (K, N) per resource block
    signal = rx[serving, np.arange(n)]
    other = rx.copy()
    other[serving, np.arange(n)] = 0.0

    load = np.ones(k)
    for _ in range(60):
        sinr = signal / (noise_mw + load @ other)
        cell_rate = spectral_efficiency(10 * np.log10(sinr)) * RB_HZ * r.rb / 1e6  # Mbit/s alone
        thr = np.zeros(n)
        new_load = np.zeros(k)
        for c in range(k):
            m = (serving == c) & (cell_rate > 0)
            if not m.any():
                continue
            need = demand[m] / cell_rate[m]
            got = _share(need)
            thr[m] = got * cell_rate[m]
            new_load[c] = got.sum()
        if np.abs(new_load - load).max() < 1e-4:
            load = new_load
            break
        load = 0.5 * load + 0.5 * new_load

    sat = np.minimum(1.0, thr / demand)
    out = {"score": float(sat.mean())}
    if detail:
        out.update(sum_thr_mbps=float(thr.sum()), p5_satisfaction=float(np.percentile(sat, 5)),
                   served=int((thr > 0).sum()), cell_load=load.tolist(), thr_mbps=thr.tolist(),
                   sinr_db=(10 * np.log10(sinr)).tolist(), serving=serving.tolist(),
                   uav_users=np.bincount(serving, minlength=k).tolist(),
                   uav_demand_mbps=np.bincount(serving, demand, minlength=k).tolist(),
                   uav_delivered_mbps=np.bincount(serving, thr, minlength=k).tolist())
    return out


FEEDBACK = ("score", "uav_users", "uav_demand_mbps", "uav_delivered_mbps")


def feedback(inst: Instance, u: np.ndarray, epoch: int = 0) -> dict:
    """What the UAVs could measure in one epoch with positions u (K, 3). No rule is checked."""
    d = _snapshot(inst, u, inst.xy(epoch), inst.demands(epoch), True)
    return {k: d[k] for k in FEEDBACK}


class Probe:
    """Blind task: the only view of the users. Each probe tries one placement on the closed-form
    scorer and returns what the UAVs themselves could measure: per UAV, how many users attached,
    how much they asked for and how much was delivered, plus the resulting score.
    An invalid placement still uses up a probe.
    """

    def __init__(self, inst: Instance):
        self._inst = inst
        self.remaining = inst.probe_budget or 0

    def __call__(self, uavs) -> dict:
        if self.remaining <= 0:
            raise RuntimeError("probe budget used up")
        self.remaining -= 1
        d = score_fast(self._inst, uavs, detail=True)
        return {k: d[k] for k in FEEDBACK}
