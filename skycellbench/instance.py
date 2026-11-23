"""Instance and placement files, the validity rules, and the seeded instance generator.

An instance fixes everything a controller cannot change: the area, the users and their demands,
the number of UAVs, the radio settings and the constraints of its task. A placement is the
controller's answer. Both are plain JSON so any kind of controller can read and write them.

Tasks (each adds one difficulty to the one before it in the list, on the same scorers):
  place        choose (x, y, altitude) per UAV; nothing else
  constrained  the same, with no-fly zones and a backhaul chain to a ground gateway
  blind        user positions are hidden; the controller gets a small number of probes
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

FORMAT_INSTANCE = "skycellbench-instance/0.2"
FORMAT_PLACEMENT = "skycellbench-placement/0.2"
TASKS = ("place", "constrained", "blind")
_SEED_KEY = {"place": 0, "constrained": 1, "blind": 3}   # fixed, so instances do not change when tasks do
GATEWAY_HEIGHT_M = 10.0
TOLERANCE_M = 0.05                 # slack on distance rules, for positions rounded in files


@dataclass
class Radio:
    freq_hz: float = 2.12e9        # LTE band 7 downlink, the ns-3 default carrier
    rb: int = 100                  # resource blocks (100 = 20 MHz); every UAV reuses the carrier
    tx_power_dbm: float = 30.0
    noise_figure_db: float = 9.0
    # Air-to-ground mean path loss (Al-Hourani et al., 2014), urban values
    a: float = 9.61
    b: float = 0.16
    eta_los_db: float = 1.0
    eta_nlos_db: float = 20.0


@dataclass
class Instance:
    id: str
    task: str
    side_m: float                  # square service area [0, side] x [0, side]
    alt_min_m: float
    alt_max_m: float
    n_uav: int
    users: list                    # [x, y, demand_mbps] per user (epoch 0); users stand at 1.5 m
    radio: Radio = field(default_factory=Radio)
    seed: int | None = None
    # constrained: circles [x, y, radius] no UAV may be above; every UAV must reach the gateway
    # through hops of at most backhaul_range_m (UAV to UAV, or UAV to gateway)
    no_fly: list = field(default_factory=list)
    gateway: list | None = None    # [x, y]
    backhaul_range_m: float | None = None
    # several epochs (no task uses this at present): user positions [x, y] per epoch (epoch 0 equals `users`) and, per epoch, 1 for the
    # users who are active (the others ask for nothing); a UAV may move at most max_move_m
    # between consecutive epochs
    tracks: list | None = None
    activity: list | None = None
    max_move_m: float | None = None
    # blind: the controller receives public() and this many probes
    probe_budget: int | None = None

    @property
    def n_epochs(self) -> int:
        return len(self.tracks) if self.tracks else 1

    @property
    def demand(self) -> np.ndarray:
        return np.asarray(self.users, float).reshape(-1, 3)[:, 2]

    def active(self, epoch: int = 0) -> np.ndarray:
        if self.activity:
            return np.asarray(self.activity[epoch], bool)
        return np.ones(len(self.users), bool)

    def xy(self, epoch: int = 0) -> np.ndarray:
        """Positions of the users active in this epoch."""
        if self.tracks:
            return np.asarray(self.tracks[epoch], float)[self.active(epoch)]
        return np.asarray(self.users, float).reshape(-1, 3)[:, :2]

    def demands(self, epoch: int = 0) -> np.ndarray:
        """Demands of the users active in this epoch."""
        return self.demand[self.active(epoch)]

    def public(self) -> "Instance":
        """What the controller is given. Only the blind task hides anything."""
        if self.task != "blind":
            return self
        d = asdict(self)
        d["users"], d["radio"] = [], Radio(**d["radio"])
        return Instance(**d)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"format": FORMAT_INSTANCE, **asdict(self)}) + "\n")

    @staticmethod
    def load(path) -> "Instance":
        d = json.loads(Path(path).read_text())
        if d.pop("format", None) != FORMAT_INSTANCE:
            raise ValueError(f"{path}: not a {FORMAT_INSTANCE} file")
        d["radio"] = Radio(**d["radio"])
        return Instance(**d)


@dataclass
class Placement:
    instance: str
    uavs: list     # [x, y, altitude] per UAV; with several epochs, one such list per epoch

    def save(self, path) -> None:
        Path(path).write_text(json.dumps({"format": FORMAT_PLACEMENT, **asdict(self)}) + "\n")

    @staticmethod
    def load(path) -> "Placement":
        d = json.loads(Path(path).read_text())
        if d.pop("format", None) != FORMAT_PLACEMENT:
            raise ValueError(f"{path}: not a {FORMAT_PLACEMENT} file")
        return Placement(**d)


def connected(inst: Instance, u: np.ndarray) -> bool:
    """True if every UAV reaches the gateway through hops of at most backhaul_range_m."""
    g = np.array([[inst.gateway[0], inst.gateway[1], GATEWAY_HEIGHT_M]])
    p = np.vstack([g, u])
    near = np.linalg.norm(p[:, None] - p[None], axis=-1) <= inst.backhaul_range_m + TOLERANCE_M
    seen = np.zeros(len(p), bool)
    seen[0] = True
    frontier = [0]
    while frontier:
        new = near[frontier.pop()] & ~seen
        seen |= new
        frontier.extend(np.flatnonzero(new).tolist())
    return bool(seen.all())


def check(inst: Instance, uavs) -> np.ndarray:
    """Return the placement as an (epochs, K, 3) array, or raise ValueError naming the rule broken."""
    u = np.asarray(uavs, float)
    if u.ndim == 2:
        u = u[None]
    if u.shape != (inst.n_epochs, inst.n_uav, 3):
        raise ValueError(f"expected shape ({inst.n_epochs}, {inst.n_uav}, 3) "
                         f"(epochs, UAVs, x/y/altitude), got {u.shape}")
    if not np.isfinite(u).all():
        raise ValueError("placement has non-finite values")
    if (u[..., :2] < 0).any() or (u[..., :2] > inst.side_m).any():
        raise ValueError("a UAV is outside the service area")
    if (u[..., 2] < inst.alt_min_m).any() or (u[..., 2] > inst.alt_max_m).any():
        raise ValueError(f"altitude must be in [{inst.alt_min_m}, {inst.alt_max_m}] m")
    for x, y, r in inst.no_fly:
        if (np.hypot(u[..., 0] - x, u[..., 1] - y) < r).any():
            raise ValueError("a UAV is inside a no-fly zone")
    if inst.gateway is not None:
        for e in range(len(u)):
            if not connected(inst, u[e]):
                raise ValueError(f"epoch {e}: a UAV has no backhaul path to the gateway")
    if inst.max_move_m is not None and len(u) > 1:
        if (np.linalg.norm(np.diff(u, axis=0), axis=-1) > inst.max_move_m + TOLERANCE_M).any():
            raise ValueError(f"a UAV moves more than {inst.max_move_m} m between epochs")
    return u


def generate(seed: int, task: str = "place", n_uav: int = 5, n_users: int = 200,
             side_m: float = 2000.0, demands=(0.5, 1.0, 2.0), demand_p=(0.4, 0.4, 0.2)) -> Instance:
    """Users are a mix of uniform background and Gaussian hotspots; demands are drawn per user.

    Hotspot share, count and spread vary with the seed so that instances differ in kind.
    """
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    rng = np.random.default_rng([seed, _SEED_KEY[task]])
    hotspot_frac = rng.uniform(0.3, 0.8)
    n_hot = int(rng.integers(2, 6))
    centres = rng.uniform(0.15 * side_m, 0.85 * side_m, size=(n_hot, 2))
    spread = rng.uniform(0.04, 0.10, size=n_hot) * side_m
    weight = rng.dirichlet(np.full(n_hot, 2.0))
    hot = np.where(rng.random(n_users) < hotspot_frac, rng.choice(n_hot, n_users, p=weight), -1)
    offset = rng.standard_normal((n_users, 2))             # position inside the hotspot
    background = rng.uniform(0, side_m, size=(n_users, 2))
    demand = rng.choice(demands, n_users, p=demand_p)

    def positions(c, bg):
        p = np.where(hot[:, None] >= 0, c[hot] + spread[hot, None] * offset, bg)
        return np.round(np.clip(p, 0, side_m), 1)

    xy = positions(centres, background)
    inst = Instance(id=f"{task}-{seed}-k{n_uav}-n{n_users}", task=task, side_m=side_m,
                    alt_min_m=50.0, alt_max_m=300.0, n_uav=n_uav, seed=seed,
                    users=[[float(x), float(y), float(d)] for (x, y), d in zip(xy, demand)])

    if task == "constrained":
        # the largest hotspots are no-fly zones, so the obvious spots are forbidden; the gateway
        # sits on an edge and the chain is too short to cover the whole area
        for h in np.argsort(-weight)[: min(3, n_hot)]:
            inst.no_fly.append([*np.round(centres[h], 1).tolist(),
                                float(np.round(rng.uniform(1.5, 2.5) * spread[h], 1))])
        while True:
            g = np.round(rng.uniform(0, side_m, 2), 1)
            g[rng.integers(2)] = rng.choice([0.0, side_m])
            if all(np.hypot(g[0] - x, g[1] - y) > r + 50 for x, y, r in inst.no_fly):
                break
        inst.gateway = g.tolist()
        inst.backhaul_range_m = float(np.round(rng.uniform(450, 650)))
    elif task == "blind":
        inst.probe_budget = 30
    return inst
