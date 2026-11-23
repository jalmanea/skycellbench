"""RL baseline: an environment in the Gymnasium style (reset/step) and a small PPO policy.

The UAVs start from the do-nothing placement and each moves a little at every step. All UAVs
share one policy, and each sees only its own surroundings: the demand around it, its own load,
its nearest neighbours and the constraints. The reward is the gain in the team score, measured on
the closed-form scorer. One policy is trained per task on freshly generated instances and then
applied, without further learning, to unseen ones.

  static tasks  29 moves; the last positions are the answer
  blind         the same, but the observation has no user map; each step uses one probe
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from .baselines import centre_placement
from .fast import Probe, epoch_mean, feedback
from .instance import Instance, connected

STEPS, PATCH, CELL_M, NEAR = 29, 7, 200.0, 3
STEP_M = np.array([200.0, 200.0, 50.0])
OBS_DIM = 11 + 4 + 2 * PATCH * PATCH + 4 * NEAR + 7 + 1
ACT_DIM = 3


def _patches(inst: Instance, epoch: int, u: np.ndarray) -> np.ndarray:
    """Demand around each UAV: a PATCH x PATCH map of CELL_M cells centred on the UAV."""
    out = np.zeros((len(u), PATCH * PATCH))
    if not inst.users:
        return out
    xy, w = inst.xy(epoch), inst.demands(epoch) / inst.demand.sum() * 10
    half = PATCH * CELL_M / 2
    for k, p in enumerate(u):
        h, _, _ = np.histogram2d(xy[:, 0] - p[0], xy[:, 1] - p[1], bins=PATCH,
                                 range=[[-half, half]] * 2, weights=w)
        out[k] = h.ravel()
    return out


class PlacementEnv:
    """`inst` is what the controller may see. On the blind task pass a Probe as well."""

    def __init__(self, inst: Instance, probe: Probe | None = None):
        self.inst, self.probe = inst, probe
        self.k = inst.n_uav
        self.lo = np.array([0, 0, inst.alt_min_m])
        self.hi = np.array([inst.side_m, inst.side_m, inst.alt_max_m])
        self.free = STEPS - (inst.n_epochs - 1)          # moves before the epochs start to run

    def _measure(self):
        if self.probe is not None:
            self.fb = self.probe(self.u)
        else:
            self.fb = feedback(self.inst, self.u, self.epoch)

    def _ok(self, u) -> bool:
        i = self.inst
        if any((np.hypot(u[:, 0] - x, u[:, 1] - y) < r).any() for x, y, r in i.no_fly):
            return False
        return i.gateway is None or connected(i, u)

    def reset(self):
        self.t, self.epoch = 0, 0
        self.u = centre_placement(self.inst)[0]
        ang = 2 * np.pi * np.arange(self.k) / self.k
        spread = np.clip(self.u + 30 * np.c_[np.cos(ang), np.sin(ang), np.zeros(self.k)],
                         self.lo, self.hi)
        if self._ok(spread):
            self.u = spread
        self.plan = []
        self.last_a, self.last_r = np.zeros((self.k, 3)), 0.0
        self._measure()
        return self._obs()

    def step(self, action):
        """`action` is (K, 3) in [-1, 1]: one move per UAV. Returns per-UAV observations."""
        a = np.clip(np.asarray(action, float).reshape(self.k, 3), -1, 1)
        i, before = self.inst, self.fb["score"]
        moving = self.t >= self.free                     # this move also advances the epoch
        if moving:
            self.plan.append(self.u.copy())
            move = a * i.max_move_m
            norm = np.linalg.norm(move, axis=1, keepdims=True)
            move *= np.minimum(1.0, (i.max_move_m - 0.01) / np.maximum(norm, 1e-9))
        else:
            move = a * STEP_M
        for k in range(self.k):                          # a move that breaks a rule is dropped
            v = self.u.copy()
            v[k] = np.clip(v[k] + move[k], self.lo, self.hi)
            if self._ok(v):
                self.u = v
        self.t += 1
        if moving:
            self.epoch += 1
        self._measure()
        if i.n_epochs == 1:
            reward = self.fb["score"] - before
        else:
            n = sum(i.active(e).sum() for e in range(i.n_epochs))
            reward = self.fb["score"] * i.active(self.epoch).sum() / n if moving else \
                (self.fb["score"] - before) * i.active(0).sum() / n
        self.last_a, self.last_r = a, reward
        done = self.t >= STEPS
        if done:
            self.plan.append(self.u.copy())
        return self._obs(), float(reward), done

    def answer(self) -> np.ndarray:
        return np.array(self.plan)

    def _obs(self) -> np.ndarray:
        """One row per UAV: what that UAV sees. All UAVs share one policy."""
        i, fb, k = self.inst, self.fb, self.k
        tot = max(sum(fb["uav_demand_mbps"]), 1e-9)
        dem, got = np.array(fb["uav_demand_mbps"]), np.array(fb["uav_delivered_mbps"])
        sat = got / np.maximum(dem, 1e-9)
        own = np.c_[(self.u - self.lo) / (self.hi - self.lo),
                    np.array(fb["uav_users"]) / max(sum(fb["uav_users"]), 1), dem / tot, sat,
                    np.tile([fb["score"], self.t / STEPS, float(self.t >= self.free), k / 8,
                             got.sum() / tot], (k, 1)),
                    self.last_a, np.full((k, 1), 10 * self.last_r)]
        nxt = min(self.epoch + 1, i.n_epochs - 1) if self.t >= self.free - 1 else self.epoch
        near = np.zeros((k, NEAR, 4))
        rel = self.u[None] - self.u[:, None]                 # rel[a, b] = b seen from a
        dist = np.linalg.norm(rel, axis=-1) + np.eye(k) * 1e9
        for a in range(k):
            for j, b in enumerate(np.argsort(dist[a])[: min(NEAR, k - 1)]):
                near[a, j] = [*np.clip(rel[a, b] / 500.0, -3, 3), sat[b]]
        con = np.zeros((k, 8))
        if i.gateway is not None:
            con[:, 0] = 1
            con[:, 1:3] = (np.array(i.gateway) - self.u[:, :2]) / i.side_m
            con[:, 3] = i.backhaul_range_m / i.side_m
        if i.no_fly:
            z = np.array(i.no_fly)
            d = self.u[:, None, :2] - z[None, :, :2]         # from zone centre to UAV
            edge = np.linalg.norm(d, axis=-1) - z[None, :, 2]
            j = edge.argmin(1)
            pick = d[np.arange(k), j]
            con[:, 4] = np.clip(edge[np.arange(k), j] / 500.0, 0, 3)
            con[:, 5:7] = pick / np.maximum(np.linalg.norm(pick, axis=1, keepdims=True), 1e-9)
        con[:, 7] = (i.max_move_m or 0) / i.side_m
        return np.hstack([own, _patches(i, self.epoch, self.u), _patches(i, nxt, self.u),
                          near.reshape(k, -1), con]).astype(np.float32)


class Policy(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        def mlp(out):
            return nn.Sequential(nn.Linear(OBS_DIM, hidden), nn.Tanh(),
                                 nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, out))
        self.pi, self.v = mlp(ACT_DIM), mlp(1)
        self.log_std = nn.Parameter(torch.full((ACT_DIM,), -0.5))

    def dist(self, obs):
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())


def rl_placement(inst: Instance, policy: Policy, probe: Probe | None = None) -> np.ndarray:
    """Roll the trained policy out once, without exploration noise, and return its answer."""
    env = PlacementEnv(inst, probe)
    obs, done = env.reset(), False
    with torch.no_grad():
        while not done:
            obs, _, done = env.step(torch.tanh(policy.pi(torch.as_tensor(obs))).numpy())
    return env.answer()


def load_policy(path) -> Policy:
    p = Policy()
    p.load_state_dict(torch.load(path))
    return p.eval()
