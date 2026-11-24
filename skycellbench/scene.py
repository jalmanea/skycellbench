"""Scenes: a real place as geometry, and the line-of-sight test the channel uses on it.

A scene file (JSON, format `skycellbench-scene/1`) holds
  crs        a local metric frame: x east, y north, z up, with the latitude and longitude of the
             point (0, 0) and the metres per degree used to project the sources
  size_m     [width, height] of the area; the area is [0, width] x [0, height]
  terrain    ground height above the vertical datum on a regular grid of nodes; row j is
             y = j * cell_m, column i is x = i * cell_m; between nodes it is bilinear
  buildings  footprints with one height each: `rings` is the outer ring followed by any holes,
             each a list of [x, y]; the building stands on the ground and is `height_m` tall
  channel_cell_m   cell size of the surface raster the channel is defined on (below)
  water      optional areas (rings, as for buildings) where no user stands; not part of the
             channel
  display    optional layers that are only drawn (roads, water, parks); nothing is computed
             from them
  sources    where every layer came from, with licence and date

The channel does not use the polygons directly. Loading a scene builds one raster of square
cells: a cell is part of a building when its centre is inside the footprint (even-odd rule over
the rings), and its surface height is the ground at its centre plus the height of the tallest
building covering it. Line of sight is then defined on that raster:

  the straight line from a user to a UAV is sampled every `channel_cell_m` of horizontal
  distance, starting one step from the user and stopping before the UAV; the link has line of
  sight when the line is at or above the surface of the cell under every sample.

This definition is what both scorers and the viewer share. `Scene.los` skips, in blocks, the
parts of a line that are certainly clear, and gives the same answer as the plain loop
`Scene.los_reference`.
"""
from __future__ import annotations

import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

FORMAT_SCENE = "skycellbench-scene/1"
SCENES = Path(os.environ.get("SKYCELLBENCH_SCENES",
                             Path(__file__).resolve().parent.parent / "scenes"))
_BLOCK = 16                        # samples per block in Scene.los; also cells per coarse tile


class Scene:
    def __init__(self, data: dict):
        if data.get("format") != FORMAT_SCENE:
            raise ValueError(f"not a {FORMAT_SCENE} file")
        self.data = data
        self.id = data["id"]
        self.size = tuple(float(v) for v in data["size_m"])
        t = data["terrain"]
        self.terrain_cell = float(t["cell_m"])
        self.terrain = np.asarray(t["z_m"], float)
        ny, nx = self.terrain.shape
        if (nx, ny) != (t["nx"], t["ny"]) or self.terrain.ndim != 2:
            raise ValueError("terrain grid does not have the stated shape")
        if not np.isfinite(self.terrain).all():
            raise ValueError("terrain has non-finite heights")
        if ((nx - 1) * self.terrain_cell < self.size[0] - 1e-6
                or (ny - 1) * self.terrain_cell < self.size[1] - 1e-6):
            raise ValueError("terrain grid does not cover the area")
        self.cell = float(data["channel_cell_m"])
        if self.cell <= 0:
            raise ValueError("channel_cell_m must be positive")
        self.buildings = data["buildings"]
        for b in self.buildings:
            if not b["height_m"] > 0:
                raise ValueError(f"building {b.get('id')}: height must be positive")
            if not b["rings"] or any(len(r) < 3 for r in b["rings"]):
                raise ValueError(f"building {b.get('id')}: a ring needs at least 3 points")
        self._rasterise()
        self._cache: dict = {}

    # ---- files -------------------------------------------------------------------------

    @staticmethod
    def load(path) -> "Scene":
        return Scene(json.loads(Path(path).read_text()))

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.data, separators=(",", ":")) + "\n")

    # ---- geometry ----------------------------------------------------------------------

    def ground(self, xy) -> np.ndarray:
        """Ground height at points (..., 2), bilinear between terrain nodes."""
        p = np.asarray(xy, float)
        ny, nx = self.terrain.shape
        fx = np.clip(p[..., 0] / self.terrain_cell, 0, nx - 1)
        fy = np.clip(p[..., 1] / self.terrain_cell, 0, ny - 1)
        ix = np.minimum(fx.astype(int), nx - 2)
        iy = np.minimum(fy.astype(int), ny - 2)
        ax, ay = fx - ix, fy - iy
        z = self.terrain
        return ((1 - ay) * ((1 - ax) * z[iy, ix] + ax * z[iy, ix + 1])
                + ay * ((1 - ax) * z[iy + 1, ix] + ax * z[iy + 1, ix + 1]))

    def _cells(self, xy):
        p = np.asarray(xy, float)
        ix = np.clip((p[..., 0] / self.cell).astype(int), 0, self.height.shape[1] - 1)
        iy = np.clip((p[..., 1] / self.cell).astype(int), 0, self.height.shape[0] - 1)
        return iy, ix

    def building_height(self, xy) -> np.ndarray:
        """Height of the building covering each point (..., 2); 0 where there is none."""
        return self.height[self._cells(xy)]

    def outdoors(self, xy) -> np.ndarray:
        return self.building_height(xy) == 0

    def standable(self, xy) -> np.ndarray:
        """True where a user can stand: not inside a building footprint and not on water."""
        return ~self.blocked[self._cells(xy)]

    def _fill(self, rings):
        """Cells whose centre is inside the rings (even-odd rule, so holes are outside)."""
        c = self.cell
        ny, nx = self._shape
        rings = [np.asarray(r, float) for r in rings]
        pts = np.vstack(rings)
        i0 = max(int(np.floor(pts[:, 0].min() / c)), 0)
        i1 = min(int(np.ceil(pts[:, 0].max() / c)), nx)
        j0 = max(int(np.floor(pts[:, 1].min() / c)), 0)
        j1 = min(int(np.ceil(pts[:, 1].max() / c)), ny)
        if i0 >= i1 or j0 >= j1:
            return None
        x = (np.arange(i0, i1) + 0.5) * c
        y = (np.arange(j0, j1) + 0.5) * c
        inside = np.zeros((j1 - j0, i1 - i0), bool)
        for r in rings:
            for (xa, ya), (xb, yb) in zip(r, np.roll(r, -1, axis=0)):
                if ya == yb:
                    continue
                rows = (ya > y) != (yb > y)
                if rows.any():
                    xc = xa + (y[rows] - ya) * (xb - xa) / (yb - ya)
                    inside[rows] ^= x[None, :] < xc[:, None]
        return slice(j0, j1), slice(i0, i1), inside

    def _rasterise(self) -> None:
        c = self.cell
        nx, ny = int(np.ceil(self.size[0] / c)), int(np.ceil(self.size[1] / c))
        self._shape = (ny, nx)
        h = np.zeros((ny, nx))
        for b in self.buildings:
            f = self._fill(b["rings"])
            if f is not None:
                part = h[f[0], f[1]]
                np.maximum(part, np.where(f[2], b["height_m"], 0.0), out=part)
        self.height = h
        self.blocked = h > 0
        for w in self.data.get("water", []):
            f = self._fill(w["rings"])
            if f is not None:
                self.blocked[f[0], f[1]] |= f[2]
        cx, cy = np.meshgrid((np.arange(nx) + 0.5) * c, (np.arange(ny) + 0.5) * c)
        self.surface = self.ground(np.stack([cx, cy], -1)) + h
        # coarse tiles for Scene.los: the highest surface in each tile and its 8 neighbours
        ty, tx = -(-ny // _BLOCK), -(-nx // _BLOCK)
        pad = np.full((ty * _BLOCK, tx * _BLOCK), -np.inf)
        pad[:ny, :nx] = self.surface
        tile = pad.reshape(ty, _BLOCK, tx, _BLOCK).max(axis=(1, 3))
        wide = np.pad(tile, 1, constant_values=-np.inf)
        self._tile_max = np.max([wide[1 + dy: 1 + dy + ty, 1 + dx: 1 + dx + tx]
                                 for dy in (-1, 0, 1) for dx in (-1, 0, 1)], axis=0)

    # ---- line of sight -----------------------------------------------------------------

    def los_reference(self, uav, users) -> np.ndarray:
        """The definition, as a plain loop: True per user who sees the UAV at `uav` (x, y, z).

        `users` is (N, 3); all heights are above the vertical datum, not above ground.
        """
        uav, users = np.asarray(uav, float), np.asarray(users, float).reshape(-1, 3)
        out = np.ones(len(users), bool)
        for j, p in enumerate(users):
            d = uav - p
            dist = float(np.hypot(d[0], d[1]))
            for i in range(1, int(np.ceil(dist / self.cell))):
                q = p + d * (i * self.cell / dist)
                if q[2] < self.surface[self._cells(q[:2])]:
                    out[j] = False
                    break
        return out

    def los(self, uav, users) -> np.ndarray:
        """True per user who sees the UAV. Same answer as `los_reference`, vectorised."""
        uav = np.ascontiguousarray(uav, float)
        users = np.ascontiguousarray(users, float).reshape(-1, 3)
        key = (uav.tobytes(), hashlib.blake2b(users.tobytes(), digest_size=16).digest())
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        c = self.cell
        d = uav[None, :] - users
        dist = np.hypot(d[:, 0], d[:, 1])
        m = np.maximum(np.ceil(dist / c).astype(np.int64) - 1, 0)       # samples per user
        step = d * (c / np.maximum(dist, 1e-12))[:, None]               # one sample step, 3-D
        out = np.ones(len(users), bool)

        def test(o, i):                     # samples number i of users o: mark the blocked
            p = users[o] + i[:, None] * step[o]
            out[o[p[:, 2] < self.surface[self._cells(p[:, :2])]]] = False

        # pass 1: the first block of every line. Most blocked links are blocked close to the
        # user, where the line is low, and need nothing more.
        i = np.broadcast_to(np.arange(1, _BLOCK + 1), (len(users), _BLOCK))
        o = np.broadcast_to(np.arange(len(users))[:, None], i.shape)
        keep = i <= m[:, None]
        test(o[keep], i[keep])
        # pass 2: the later blocks of the lines still clear. A block of _BLOCK consecutive
        # samples spans at most one tile width, so it stays within the tile under its first
        # point and that tile's 8 neighbours; if the line is above all of them, it is clear.
        left = np.flatnonzero(out & (m > _BLOCK))
        nb = -(-m[left] // _BLOCK) - 1
        owner = np.repeat(left, nb)
        lo = (np.arange(nb.sum()) - np.repeat(np.cumsum(nb) - nb, nb) + 1) * _BLOCK
        hi = np.minimum(lo + _BLOCK, m[owner])           # block = samples lo+1 .. hi
        zmin = users[owner, 2] + np.minimum(lo * step[owner, 2], hi * step[owner, 2])
        start = users[owner, :2] + lo[:, None] * step[owner, :2]
        ty, tx = self._tile_max.shape
        tix = np.clip((start[:, 0] / (c * _BLOCK)).astype(int), 0, tx - 1)
        tiy = np.clip((start[:, 1] / (c * _BLOCK)).astype(int), 0, ty - 1)
        maybe = np.flatnonzero(zmin < self._tile_max[tiy, tix])
        if len(maybe):
            i = lo[maybe, None] + np.arange(1, _BLOCK + 1)[None, :]
            o = np.broadcast_to(owner[maybe, None], i.shape)
            keep = i <= m[o]
            test(o[keep], i[keep])
        if len(self._cache) >= 20_000:               # a search asks again for the UAVs it did
            self._cache.clear()                      # not move; old entries are of no use
        self._cache[key] = out
        return out


@lru_cache(maxsize=8)
def get_scene(scene_id: str) -> Scene:
    """Load the scene `scenes/<id>/scene.json` once per process."""
    path = SCENES / scene_id / "scene.json"
    if not path.exists():
        raise FileNotFoundError(f"scene '{scene_id}' not found at {path}")
    s = Scene.load(path)
    if s.id != scene_id:
        raise ValueError(f"{path} holds scene '{s.id}', not '{scene_id}'")
    return s
