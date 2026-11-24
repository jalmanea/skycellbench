"""Build scenes/cincinnati-downtown/scene.json from public sources.

  python scenes/cincinnati-downtown/build.py              use the files in raw/ (download if absent)
  python scenes/cincinnati-downtown/build.py --download   fetch everything again

Sources (details, licences and dates are written into the scene and listed in SOURCES.md):
  buildings, roads, water, parks   OpenStreetMap, through an Overpass API server
  ground height                    USGS 3D Elevation Program, bare-earth elevation service
  ten tower heights, as a fallback DroneRTS city-research.json (documented architectural heights)

The area is 2 km by 2 km. Its south-west corner is ORIGIN; it contains the 820 m by 660 m area of
DroneRTS. Coordinates are a local equirectangular projection about ORIGIN: over 2 km it differs
from a true metric projection by a few centimetres.
"""
import datetime
import hashlib
import json
import math
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
ORIGIN_LAT, ORIGIN_LON = 39.0961, -84.5225         # south-west corner of the area
SIZE_M = 2000.0
TERRAIN_CELL_M = 10.0
CHANNEL_CELL_M = 2.0
M_PER_LEVEL = 4.0                                   # median of height / levels where OSM has both
SMALL = {"garage", "garages", "shed", "carport", "roof", "service", "kiosk", "hut"}
SMALL_M, DEFAULT_LEVELS = 3.0, 2                    # buildings with no height information
OVERPASS = ["https://overpass-api.de/api/interpreter",
            "https://maps.mail.ru/osm/tools/overpass/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter"]
USGS = "https://elevation.nationalmap.gov/arcgis/rest/services/3DEPElevation/ImageServer"
AGENT = {"User-Agent": "scene-builder/0.1"}
# Documented architectural heights collected by DroneRTS (CINCINNATI-SOURCES.md, MIT), keyed by
# OSM way id. Used only where OpenStreetMap has no height tag.
DRONERTS_DOCUMENTED = {46126970: 150.9, 46129991: 127.4, 91327377: 113.4, 91493361: 202.7,
                       100146895: 175.0, 39197415: 142.7, 28719762: 129.0, 46703730: 125.0,
                       28719627: 112.2, 28719652: 108.0}
# DroneRTS massing estimates for large venues that OpenStreetMap gives no height or levels for.
DRONERTS_ESTIMATED = {24587940: 38.0, 210540615: 48.0, 25564082: 30.0, 30736346: 22.0}

phi = math.radians(ORIGIN_LAT + SIZE_M / 2 / 111000)          # mid-latitude of the area
M_PER_DEG_LAT = 111132.92 - 559.82 * math.cos(2 * phi) + 1.175 * math.cos(4 * phi)
M_PER_DEG_LON = 111412.84 * math.cos(phi) - 93.5 * math.cos(3 * phi) + 0.118 * math.cos(5 * phi)


def to_xy(lat, lon):
    return ((lon - ORIGIN_LON) * M_PER_DEG_LON, (lat - ORIGIN_LAT) * M_PER_DEG_LAT)


def to_latlon(x, y):
    return (ORIGIN_LAT + y / M_PER_DEG_LAT, ORIGIN_LON + x / M_PER_DEG_LON)


# ---- downloads -------------------------------------------------------------------------------

def bbox(margin_m):
    s, w = to_latlon(-margin_m, -margin_m)
    n, e = to_latlon(SIZE_M + margin_m, SIZE_M + margin_m)
    return s, w, n, e


def overpass(query, path):
    for attempt in range(12):                        # busy servers are common: go round them
        url = OVERPASS[attempt % len(OVERPASS)]
        try:
            req = urllib.request.Request(url, urllib.parse.urlencode({"data": query}).encode(),
                                         AGENT)
            body = urllib.request.urlopen(req, timeout=240).read()
            json.loads(body)["elements"]
        except Exception as e:
            print("  ", url, "failed:", str(e)[:80], flush=True)
            time.sleep(20 * (attempt // len(OVERPASS) + 1))
            continue
        path.write_bytes(body)
        (path.parent / (path.name + ".meta")).write_text(json.dumps(
            {"url": url, "query": query, "retrieved": datetime.date.today().isoformat()}))
        return
    raise SystemExit("no Overpass server answered")


def download():
    RAW.mkdir(exist_ok=True)
    s, w, n, e = bbox(300)
    box = f"[bbox:{s:.5f},{w:.5f},{n:.5f},{e:.5f}]"
    overpass(f"[out:json][timeout:180]{box};(way[\"building\"];"
             "relation[\"building\"][\"type\"=\"multipolygon\"];way[\"building:part\"];"
             "relation[\"building:part\"][\"type\"=\"multipolygon\"];);out body geom;",
             RAW / "osm-buildings.json")
    s, w, n, e = bbox(50)
    box = f"[bbox:{s:.5f},{w:.5f},{n:.5f},{e:.5f}]"
    roads = ("motorway|trunk|primary|secondary|tertiary|residential|unclassified|living_street|"
             "pedestrian|motorway_link|trunk_link|primary_link|secondary_link|tertiary_link")
    overpass(f"[out:json][timeout:180]{box};(way[\"highway\"~\"^({roads})$\"];"
             "way[\"natural\"=\"water\"];relation[\"natural\"=\"water\"];"
             "way[\"leisure\"=\"park\"];relation[\"leisure\"=\"park\"];);out body geom;",
             RAW / "osm-display.json")
    # elevation: 2 m pixels whose 5 x 5 blocks are centred on the terrain nodes
    half = TERRAIN_CELL_M / 2
    s, w = to_latlon(-half, -half)
    n, e = to_latlon(SIZE_M + half, SIZE_M + half)
    px = int(round((SIZE_M + TERRAIN_CELL_M) / 2.0))
    q = urllib.parse.urlencode({
        "bbox": f"{w:.7f},{s:.7f},{e:.7f},{n:.7f}", "bboxSR": 4326, "imageSR": 4326,
        "size": f"{px},{px}", "format": "tiff", "pixelType": "F32",
        "interpolation": "RSP_BilinearInterpolation", "f": "image"})
    url = f"{USGS}/exportImage?{q}"
    body = urllib.request.urlopen(urllib.request.Request(url, headers=AGENT), timeout=240).read()
    (RAW / "usgs-3dep.tif").write_bytes(body)
    (RAW / "usgs-3dep.tif.meta").write_text(json.dumps(
        {"url": url, "retrieved": datetime.date.today().isoformat()}))


# ---- geometry helpers ------------------------------------------------------------------------

def area(ring):
    r = np.asarray(ring)
    return 0.5 * float(np.sum(r[:, 0] * np.roll(r[:, 1], -1) - np.roll(r[:, 0], -1) * r[:, 1]))


def clip_ring(ring, lo=0.0, hi=SIZE_M):
    """Sutherland-Hodgman clip of a closed ring to the square area."""
    pts = list(ring)
    for axis, bound, keep_above in ((0, lo, True), (0, hi, False), (1, lo, True), (1, hi, False)):
        out = []
        for a, b in zip(pts, pts[1:] + pts[:1]):
            ia = a[axis] >= bound if keep_above else a[axis] <= bound
            ib = b[axis] >= bound if keep_above else b[axis] <= bound
            if ia != ib:
                t = (bound - a[axis]) / (b[axis] - a[axis])
                cut = [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]
                cut[axis] = bound
                if ia:
                    out.append(cut)
                else:
                    out += [cut, b]
            elif ib:
                out.append(b)
        pts = out
        if len(pts) < 3:
            return []
    return pts


def clip_line(points, lo=0.0, hi=SIZE_M):
    """Parts of a polyline inside the area, as a list of polylines."""
    parts, cur = [], []
    for a, b in zip(points, points[1:]):
        t0, t1 = 0.0, 1.0
        d = (b[0] - a[0], b[1] - a[1])
        ok = True
        for p, q in ((-d[0], a[0] - lo), (d[0], hi - a[0]), (-d[1], a[1] - lo), (d[1], hi - a[1])):
            if p == 0:
                ok &= q >= 0
            elif p < 0:
                t0 = max(t0, q / p)
            else:
                t1 = min(t1, q / p)
        if not ok or t0 > t1:
            if len(cur) > 1:
                parts.append(cur)
            cur = []
            continue
        pa = [a[0] + t0 * d[0], a[1] + t0 * d[1]]
        pb = [a[0] + t1 * d[0], a[1] + t1 * d[1]]
        if not cur or t0 > 0:
            if len(cur) > 1:
                parts.append(cur)
            cur = [pa]
        cur.append(pb)
        if t1 < 1:
            parts.append(cur)
            cur = []
    if len(cur) > 1:
        parts.append(cur)
    return parts


def inside(pt, ring):
    x, y = pt
    r = np.asarray(ring)
    a, b = r, np.roll(r, -1, axis=0)
    cross = (a[:, 1] > y) != (b[:, 1] > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = a[:, 0] + (y - a[:, 1]) * (b[:, 0] - a[:, 0]) / (b[:, 1] - a[:, 1])
    return bool(np.sum(cross & (x < xc)) % 2)


def rings_of(el):
    """Outer and inner rings of a way or multipolygon relation, in metres. Open ways are joined."""
    def xy(geom):
        return [list(to_xy(p["lat"], p["lon"])) for p in geom]
    if el["type"] == "way":
        g = xy(el.get("geometry", []))
        return ([g[:-1]] if len(g) > 3 and g[0] == g[-1] else []), []
    out = {"outer": [], "inner": []}
    for role in out:
        ways = [xy(m["geometry"]) for m in el["members"]
                if m["type"] == "way" and m.get("role") == role and "geometry" in m]
        while ways:
            ring = ways.pop()
            while ring[0] != ring[-1]:
                for i, w in enumerate(ways):
                    if w[0] == ring[-1]:
                        ring += ways.pop(i)[1:]
                        break
                    if w[-1] == ring[-1]:
                        ring += ways.pop(i)[::-1][1:]
                        break
                else:
                    ring = None
                    break
            if ring and len(ring) > 3:
                out[role].append(ring[:-1])
    return out["outer"], out["inner"]


def number(text):
    try:
        v = float(str(text).split()[0].replace(",", "."))
        return v if math.isfinite(v) and v > 0 else None
    except (ValueError, IndexError):
        return None


def rnd(ring, nd=2):
    return [[round(x, nd), round(y, nd)] for x, y in ring]


# ---- build -----------------------------------------------------------------------------------

def build():
    osm = json.loads((RAW / "osm-buildings.json").read_text())
    els = osm["elements"]
    parts = []                                       # (centroid, height) of building parts
    for e in els:
        t = e.get("tags", {})
        if "building:part" in t and "building" not in t:
            h = number(t.get("height")) or (number(t.get("building:levels")) or 0) * M_PER_LEVEL
            outer, _ = rings_of(e)
            if h and outer:
                parts.append((np.mean(outer[0], axis=0), h))

    buildings, how, skipped = [], Counter(), Counter()
    for e in els:
        t = e.get("tags", {})
        if "building" not in t or t["building"] == "no":
            continue
        if t.get("location") == "underground" or str(t.get("layer", "0")).strip().startswith("-"):
            skipped["underground"] += 1
            continue
        outer, inner = rings_of(e)
        if not outer:
            skipped["no closed outline"] += 1
            continue
        oid = e["id"]
        levels = number(t.get("building:levels"))
        part_h = [h for c, h in parts if any(inside(c, o) for o in outer)]
        if number(t.get("height")):
            h, src = number(t["height"]), "osm:height"
        elif e["type"] == "way" and oid in DRONERTS_DOCUMENTED:
            h, src = DRONERTS_DOCUMENTED[oid], "dronerts:documented"
        elif part_h:
            h, src = max(part_h), "osm:building:part"
        elif levels:
            h, src = levels * M_PER_LEVEL, "osm:levels"
        elif e["type"] == "way" and oid in DRONERTS_ESTIMATED:
            h, src = DRONERTS_ESTIMATED[oid], "dronerts:estimated"
        elif t["building"] in SMALL:
            h, src = SMALL_M, "default:small"
        else:
            h, src = DEFAULT_LEVELS * M_PER_LEVEL, "default"
        for k, o in enumerate(outer):
            holes = [i for i in inner if inside(i[0], o)]
            rings = [r for r in (clip_ring(o), *map(clip_ring, holes)) if len(r) >= 3]
            if not rings or len(clip_ring(o)) < 3 or abs(area(rings[0])) < 4.0:
                skipped["outside the area or under 4 m2"] += 1
                continue
            b = {"id": f"osm-{e['type']}-{oid}" + (f"-{k}" if len(outer) > 1 else ""),
                 "height_m": round(h, 1), "height_source": src, "rings": [rnd(r) for r in rings]}
            if t.get("name"):
                b["name"] = t["name"]
            buildings.append(b)
            how[src] += 1
    buildings.sort(key=lambda b: b["id"])

    # terrain: block means of the 2 m elevation raster, row 0 at the south edge
    from PIL import Image
    dem = np.asarray(Image.open(RAW / "usgs-3dep.tif"), float)[::-1]
    n = int(round(SIZE_M / TERRAIN_CELL_M)) + 1
    f = dem.shape[0] // n
    if dem.shape != (n * f, n * f) or not np.isfinite(dem).all() or dem.min() < -100:
        raise SystemExit(f"unexpected elevation raster {dem.shape}")
    z = dem.reshape(n, f, n, f).mean(axis=(1, 3))

    disp = json.loads((RAW / "osm-display.json").read_text())
    roads, water, parks = [], [], []
    for e in disp["elements"]:
        t = e.get("tags", {})
        if "highway" in t and e["type"] == "way":
            if t.get("tunnel") in ("yes", "building_passage"):
                continue
            line = [list(to_xy(p["lat"], p["lon"])) for p in e["geometry"]]
            for part in clip_line(line):
                r = {"kind": t["highway"], "points": rnd(part, 1)}
                if t.get("name"):
                    r["name"] = t["name"]
                if t.get("bridge") and t["bridge"] != "no":
                    r["bridge"] = True
                roads.append(r)
        elif t.get("natural") == "water" or t.get("leisure") == "park":
            outer, inner = rings_of(e)
            for o in outer:
                rings = [r for r in (clip_ring(o), *(clip_ring(i) for i in inner
                                                     if inside(i[0], o))) if len(r) >= 3]
                if rings and len(clip_ring(o)) >= 3 and abs(area(rings[0])) >= 20.0:
                    item = {"id": f"osm-{e['type']}-{e['id']}", "rings": [rnd(r, 1) for r in rings]}
                    if t.get("name"):
                        item["name"] = t["name"]
                    (water if t.get("natural") == "water" else parks).append(item)

    def meta(name):
        m = json.loads((RAW / (name + ".meta")).read_text())
        m["sha256"] = hashlib.sha256((RAW / name).read_bytes()).hexdigest()
        return m
    mb, md, mt = meta("osm-buildings.json"), meta("osm-display.json"), meta("usgs-3dep.tif")
    odbl = "Open Database License (ODbL) 1.0, https://opendatacommons.org/licenses/odbl/1-0/"
    scene = {
        "format": "skycellbench-scene/1",
        "id": "cincinnati-downtown",
        "name": "Downtown Cincinnati and Over-the-Rhine, Ohio, USA",
        "crs": {"type": "local equirectangular", "origin_lat": ORIGIN_LAT,
                "origin_lon": ORIGIN_LON, "m_per_deg_lat": round(M_PER_DEG_LAT, 3),
                "m_per_deg_lon": round(M_PER_DEG_LON, 3),
                "axes": "x east, y north, z up, metres; (0, 0) is the south-west corner",
                "vertical_datum": "NAVD88"},
        "size_m": [SIZE_M, SIZE_M],
        "channel_cell_m": CHANNEL_CELL_M,
        "terrain": {"cell_m": TERRAIN_CELL_M, "nx": n, "ny": n,
                    "z_m": np.round(z, 2).tolist()},
        "buildings": buildings,
        "water": water,
        "display": {"roads": roads, "parks": parks},
        "attribution": "Map data © OpenStreetMap contributors (ODbL 1.0). Elevation: U.S. "
                       "Geological Survey, 3D Elevation Program.",
        "sources": [
            {"layer": "buildings", "source": "OpenStreetMap, Overpass API extract",
             "licence": odbl, "attribution": "© OpenStreetMap contributors",
             "url": mb["url"], "query": mb["query"], "retrieved": mb["retrieved"],
             "osm_base": osm["osm3s"]["timestamp_osm_base"], "sha256": mb["sha256"],
             "height_sources": dict(sorted(how.items())),
             "note": f"Heights: OSM height tag; else a documented tower height from DroneRTS; "
                     f"else the tallest OSM building:part inside the outline; else "
                     f"building:levels x {M_PER_LEVEL} m; else a DroneRTS estimate for four "
                     f"venues; else {SMALL_M} m for garages, sheds and roofs and "
                     f"{DEFAULT_LEVELS * M_PER_LEVEL} m for everything else. Underground "
                     f"buildings are left out; min_height is ignored, so every building is "
                     f"solid down to the ground."},
            {"layer": "building heights (fallback for 14 buildings at most)",
             "source": "DroneRTS, city-research.json and CINCINNATI-SOURCES.md",
             "licence": "MIT (code and notes); the heights are published facts from the "
                        "Skyscraper Center and Thornton Tomasetti, or DroneRTS estimates",
             "url": "https://github.com/DanMcInerney/DroneRTS",
             "commit": "0dcf03f4e1e61ac2fdfe89d3e4cbc6a89380499e"},
            {"layer": "terrain", "source": "U.S. Geological Survey, 3D Elevation Program "
                                           "(3DEP) bare-earth elevation service",
             "licence": "Public domain (U.S. Government work); credit: U.S. Geological Survey",
             "url": mt["url"], "retrieved": mt["retrieved"], "sha256": mt["sha256"],
             "note": f"Requested at 2 m and averaged in {f} x {f} blocks to a "
                     f"{TERRAIN_CELL_M:.0f} m grid. Bare earth: no buildings, bridges or "
                     f"trees."},
            {"layer": "water, display.roads, display.parks",
             "source": "OpenStreetMap, Overpass API extract", "licence": odbl,
             "attribution": "© OpenStreetMap contributors", "url": md["url"],
             "query": md["query"], "retrieved": md["retrieved"],
             "osm_base": disp["osm3s"]["timestamp_osm_base"], "sha256": md["sha256"]},
        ],
    }
    (HERE / "scene.json").write_text(json.dumps(scene, separators=(",", ":")) + "\n")
    hs = np.array([b["height_m"] for b in buildings])
    print(f"{len(buildings)} buildings ({dict(how)}), skipped {dict(skipped)}")
    print(f"heights: median {np.median(hs):.1f} m, max {hs.max():.1f} m, "
          f"{(hs > 50).sum()} above 50 m")
    print(f"terrain {z.min():.1f} to {z.max():.1f} m; {len(roads)} road parts, "
          f"{len(water)} water areas, {len(parks)} parks")
    print(f"scene.json: {(HERE / 'scene.json').stat().st_size / 1e6:.2f} MB")


if __name__ == "__main__":
    if "--download" in sys.argv or not (RAW / "usgs-3dep.tif").exists():
        download()
    build()
