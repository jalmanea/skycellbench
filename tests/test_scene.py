import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

import skycellbench as ub
from skycellbench import scene as scene_mod
from skycellbench.baselines import centre_placement, kmeans_placement, local_search, random_placement
from skycellbench.fast import UE_HEIGHT_M, path_loss_db
from skycellbench.ns3 import write_scenario
from skycellbench.scene import FORMAT_SCENE, Scene

CITY = "cincinnati-downtown"


def toy(slope=0.0, buildings=None, water=None, size=200.0):
    """A small scene: ground rising `slope` metres per metre towards the east."""
    n = int(size / 10) + 1
    z = [[100.0 + slope * 10 * i for i in range(n)] for _ in range(n)]
    return {"format": FORMAT_SCENE, "id": "toy", "size_m": [size, size], "channel_cell_m": 2.0,
            "crs": {"type": "local equirectangular", "origin_lat": 0.0, "origin_lon": 0.0},
            "terrain": {"cell_m": 10.0, "nx": n, "ny": n, "z_m": z},
            "buildings": buildings if buildings is not None else
            [{"id": "wall", "height_m": 20.0, "rings": [[[40, 0], [60, 0], [60, 100], [40, 100]]]}],
            "water": water or [], "sources": []}


@pytest.fixture
def toy_dir(tmp_path, monkeypatch):
    """Scenes are found by id in a directory; point it at a temporary one holding `toy`."""
    def install(**kw):
        (tmp_path / "toy").mkdir(exist_ok=True)
        (tmp_path / "toy" / "scene.json").write_text(json.dumps(toy(**kw)))
        scene_mod.get_scene.cache_clear()
        return scene_mod.get_scene("toy")
    monkeypatch.setattr(scene_mod, "SCENES", tmp_path)
    yield install
    scene_mod.get_scene.cache_clear()


def toy_instance(n_uav=1, users=((10.0, 50.0, 1.0),)):
    return ub.Instance(id="t", task="place", side_m=200.0, alt_min_m=5.0, alt_max_m=150.0,
                       n_uav=n_uav, users=[list(u) for u in users], scene="toy")


# ---- the scene format ---------------------------------------------------------------------

def test_scene_round_trips_and_rejects_bad_files(tmp_path):
    s = Scene(toy())
    s.save(tmp_path / "s.json")
    assert Scene.load(tmp_path / "s.json").data == s.data
    for change in ({"format": "skycellbench-scene/0"}, {"channel_cell_m": 0},
                   {"terrain": {"cell_m": 10.0, "nx": 5, "ny": 21, "z_m": toy()["terrain"]["z_m"]}},
                   {"terrain": {"cell_m": 5.0, "nx": 21, "ny": 21, "z_m": toy()["terrain"]["z_m"]}},
                   {"buildings": [{"id": "b", "height_m": 0.0, "rings": [[[0, 0], [9, 0], [9, 9]]]}]},
                   {"buildings": [{"id": "b", "height_m": 5.0, "rings": [[[0, 0], [9, 0]]]}]}):
        with pytest.raises(ValueError):
            Scene({**toy(), **change})


def test_ground_is_bilinear_and_altitude_is_above_ground(toy_dir):
    s = toy_dir(slope=0.1, buildings=[])
    assert np.allclose(s.ground([[0, 0], [55, 120], [200, 7]]), [100.0, 105.5, 120.0])
    inst = toy_instance()
    assert np.allclose(inst.absolute([[150.0, 20.0, 30.0]]), [[150.0, 20.0, 145.0]])
    flat = dataclasses.replace(inst, scene=None)
    assert np.allclose(flat.absolute([[150.0, 20.0, 30.0]]), [[150.0, 20.0, 30.0]])


def test_footprints_holes_and_water_on_the_raster():
    court = {"id": "court", "height_m": 12.0,
             "rings": [[[100, 100], [160, 100], [160, 160], [100, 160]],
                       [[120, 120], [140, 120], [140, 140], [120, 140]]]}
    pond = {"id": "pond", "rings": [[[10, 150], [30, 150], [30, 170], [10, 170]]]}
    s = Scene(toy(buildings=[court], water=[pond]))
    pts = np.array([[110, 110], [130, 130], [90, 110], [20, 160]])     # wall, court, street, pond
    assert s.building_height(pts).tolist() == [12.0, 0.0, 0.0, 0.0]
    assert s.outdoors(pts).tolist() == [False, True, True, True]
    assert s.standable(pts).tolist() == [False, True, True, False]
    assert np.isclose(s.surface[55, 55], 112.0) and np.isclose(s.surface[65, 65], 100.0)


# ---- line of sight ------------------------------------------------------------------------

def test_a_building_blocks_the_line_only_below_its_roof():
    s = Scene(toy())
    user = [[10.0, 50.0, 100.0 + UE_HEIGHT_M]]
    # the line to a UAV at x = 90 crosses the near edge of the 20 m wall at 3/8 of the way, so
    # it clears the roof when the UAV is more than (20 - 1.5) * 8 / 3 = 49.3 m above the user
    assert s.los([90.0, 50.0, 100.0 + 53.0], user).tolist() == [True]
    assert s.los([90.0, 50.0, 100.0 + 49.0], user).tolist() == [False]
    assert s.los([90.0, 150.0, 100.0 + 10.0], [[10.0, 150.0, 101.5]]).tolist() == [True]  # beside it
    assert s.los([30.0, 50.0, 100.0 + 5.0], user).tolist() == [True]                      # in front


def test_terrain_blocks_the_line():
    d = toy(buildings=[])
    for row in d["terrain"]["z_m"]:
        row[10] = 140.0                                               # a ridge along x = 100
    s = Scene(d)
    user = [[20.0, 100.0, 101.5]]
    assert s.los([180.0, 100.0, 130.0], user).tolist() == [False]
    assert s.los([180.0, 100.0, 190.0], user).tolist() == [True]


def test_fast_test_equals_the_definition_and_height_never_hides_a_user():
    s = ub.get_scene(CITY)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 2000, size=(400, 2))
    xy = xy[s.standable(xy)][:60]
    users = np.column_stack([xy, s.ground(xy) + UE_HEIGHT_M])
    seen = 0
    for _ in range(6):
        p = rng.uniform(0, 2000, 2)
        low = np.array([*p, s.ground(p) + max(60.0, s.building_height(p) + 5)])
        a = s.los(low, users)
        assert (a == s.los_reference(low, users)).all()
        assert (s.los(low + [0, 0, 150.0], users) >= a).all()         # higher sees at least as much
        seen += a.sum()
    assert 0 < seen < 6 * len(users)                                  # some links of each kind


def test_path_loss_takes_line_of_sight_from_the_scene(toy_dir):
    toy_dir()
    inst = toy_instance(users=[(10.0, 50.0, 1.0), (150.0, 50.0, 1.0)])
    u = np.array([[90.0, 50.0, 30.0]])                    # behind the wall for the first user
    loss = path_loss_db(inst, u, inst.xy())[0]
    dist = np.hypot([80.0, 60.0], 30.0 - UE_HEIGHT_M)
    fspl = 20 * np.log10(4 * np.pi * inst.radio.freq_hz * dist / 299792458.0)
    assert np.allclose(loss, fspl + [inst.radio.eta_nlos_db, inst.radio.eta_los_db])
    d = ub.score_fast(inst, u, detail=True)
    assert d["los"] == [False, True]


# ---- rules on a scene ---------------------------------------------------------------------

def test_a_uav_may_not_be_inside_a_building(toy_dir):
    toy_dir()
    inst = toy_instance()
    ub.check(inst, [[50.0, 50.0, 21.0]])                              # above the 20 m roof
    ub.check(inst, [[30.0, 50.0, 10.0]])                              # low, but beside it
    with pytest.raises(ValueError, match="inside a building"):
        ub.check(inst, [[50.0, 50.0, 20.0]])
    assert centre_placement(toy_instance()).shape == (1, 1, 3)


def test_baselines_lift_a_uav_out_of_a_building(toy_dir):
    toy_dir(buildings=[{"id": "tower", "height_m": 120.0,
                        "rings": [[[60, 60], [140, 60], [140, 140], [60, 140]]]}])
    inst = toy_instance()                                             # centre is inside the tower
    u = centre_placement(inst)
    ub.check(inst, u)
    assert u[0, 0, 2] > 120.0


@pytest.mark.parametrize("task", ub.TASKS)
def test_tasks_on_the_city_scene(task, tmp_path):
    s = ub.get_scene(CITY)
    inst = ub.generate(2, task, n_uav=4, n_users=80, scene=CITY)
    assert inst == ub.generate(2, task, n_uav=4, n_users=80, scene=CITY)
    assert inst.side_m == s.size[0] and s.standable(inst.xy()).all()
    plain = ub.generate(2, task, n_uav=4, n_users=80)
    assert inst.demand.tolist() == plain.demand.tolist() and inst.no_fly == plain.no_fly
    if inst.gateway is not None:
        assert s.standable(inst.gateway)
    inst.save(tmp_path / "i.json")
    assert ub.Instance.load(tmp_path / "i.json") == inst
    assert inst.public().scene == CITY
    rng = np.random.default_rng(0)
    for u in (centre_placement(inst), random_placement(inst, rng), kmeans_placement(inst, rng)):
        ub.check(inst, u)
    best = local_search(inst, rng, budget=300)
    ub.check(inst, best)
    assert ub.score_fast(inst, centre_placement(inst))["score"] < ub.score_fast(inst, best)["score"]


def test_kmeans_over_the_tallest_tower_is_still_valid():
    s = ub.get_scene(CITY)
    tower = max(s.buildings, key=lambda b: b["height_m"])
    x, y = np.mean(tower["rings"][0], axis=0)
    inst = ub.generate(2, n_uav=1, n_users=30, scene=CITY)
    inst.users = [[float(x + dx), float(y - 60.0), 1.0] for dx in (-1.0, 1.0)]   # street below
    inst.users += [[float(x + dx), float(y + 60.0), 1.0] for dx in (-1.0, 1.0)]
    u = kmeans_placement(inst, np.random.default_rng(0), altitude=100.0)
    ub.check(inst, u)
    assert u[0, 0, 2] > tower["height_m"]


def test_files_written_before_scenes_still_load():
    old = sorted(Path(__file__).resolve().parent.parent.glob("instances/public/place-*.json"))[0]
    assert '"skycellbench-instance/0.2"' in old.read_text()[:60]
    inst = ub.Instance.load(old)
    assert inst.scene is None and inst.get_scene() is None


# ---- one loss table for both scorers ------------------------------------------------------

def test_ns3_is_handed_the_loss_table_of_the_closed_form_scorer(tmp_path):
    inst = ub.generate(5, n_uav=3, n_users=25, scene=CITY)
    u = kmeans_placement(inst, np.random.default_rng(0))[0]
    write_scenario(inst, u, inst.xy(), inst.demands(), tmp_path / "s.txt")
    lines = (tmp_path / "s.txt").read_text().splitlines()
    at = lines.index("loss")
    table = np.array([[float(v) for v in row.split()] for row in lines[at + 1: at + 4]])
    assert (table == path_loss_db(inst, u, inst.xy())).all()          # exact, not rounded
    sc = ub.get_scene(CITY)
    z = float(lines[2].split()[2])
    assert np.isclose(z, u[0, 2] + sc.ground(u[0, :2]))               # heights above the datum


@pytest.mark.skipif(not ub.ns3.BINARY.exists(), reason="ns-3 scorer not built")
def test_ns3_scorer_agrees_on_a_small_scene_case():
    inst = ub.generate(5, n_uav=2, n_users=20, scene=CITY)
    u = kmeans_placement(inst, np.random.default_rng(0))
    fast = ub.score_fast(inst, u, detail=True)
    slow = ub.score_ns3(inst, u, sim_time=1.0, detail=True)
    assert slow["serving"] == fast["serving"]
    assert abs(fast["score"] - slow["score"]) < 0.1


# ---- the city scene file ------------------------------------------------------------------

def test_city_scene_records_its_sources_and_contains_the_dronerts_area():
    s = ub.get_scene(CITY)
    d = s.data
    assert d["format"] == FORMAT_SCENE and s.size == (2000.0, 2000.0)
    assert len(d["sources"]) >= 3 and all(x.get("licence") and x.get("url") for x in d["sources"])
    assert "OpenStreetMap" in d["attribution"]
    assert {b["height_source"] for b in s.buildings} <= {
        "osm:height", "osm:building:part", "osm:levels", "dronerts:documented",
        "dronerts:estimated", "default", "default:small"}
    # DroneRTS: 820 m x 660 m about 39.1015 N, 84.512 W; 320 m west, 500 m east, 200 m north and
    # 460 m south of that point, in its own metres per degree
    crs = d["crs"]
    for east, north in ((-320, -460), (500, 200)):
        lon, lat = -84.512 + east / 86387.648, 39.1015 + north / 111320.0
        x = (lon - crs["origin_lon"]) * crs["m_per_deg_lon"]
        y = (lat - crs["origin_lat"]) * crs["m_per_deg_lat"]
        assert 0 < x < 2000 and 0 < y < 2000


# ---- answers for the viewer -----------------------------------------------------------------

def test_view_saves_the_answer_with_what_each_user_received(tmp_path):
    from skycellbench.__main__ import view_result
    inst = ub.generate(5, n_uav=3, n_users=25, scene=CITY)
    u = np.round(kmeans_placement(inst, np.random.default_rng(0)), 2)
    p = ub.Placement(inst.id, u.tolist())
    p.save(tmp_path / "plain.json")
    assert '"result"' not in (tmp_path / "plain.json").read_text()    # unscored: as before
    p.result = view_result(inst, p.uavs)
    p.save(tmp_path / "scored.json")
    q = ub.Placement.load(tmp_path / "scored.json")
    assert q == p and q.result["tier"] == "fast"
    e = q.result["epochs"][0]
    assert len(e["thr_mbps"]) == len(e["serving"]) == len(e["los"]) == 25
    assert q.result["score"] == ub.score_fast(inst, q.uavs)["score"]


def test_viewer_examples_match_their_instances():
    folder = Path(__file__).resolve().parent.parent / "viewer" / "examples"
    index = json.loads((folder / "index.json").read_text())
    assert index
    for item in index:
        inst = ub.Instance.load(folder / item["instance"])
        answer = ub.Placement.load(folder / item["answer"])
        assert answer.instance == inst.id
        ub.check(inst, answer.uavs)
        assert np.isclose(answer.result["score"], ub.score_fast(inst, answer.uavs)["score"])


@pytest.mark.parametrize("name,base,scene", [
    ("public", 100000, None), ("heldout", 200000, None),
    (f"public-{CITY}", 100000, CITY), (f"heldout-{CITY}", 200000, CITY)])
def test_instance_sets_are_what_the_generator_writes(name, base, scene, tmp_path):
    """Every set on disk is regenerated from its seeds and compared, both written in the current
    format (the flat sets are older files). Nothing is scored: held-out files are only loaded."""
    files = sorted((Path(__file__).resolve().parent.parent / "instances" / name).glob("*.json"))
    per_task = 30 if name.startswith("heldout") else 10
    assert len(files) == 3 * per_task
    for f in files:
        inst = ub.Instance.load(f)
        assert inst.scene == scene
        seed = int(f.stem.split("-")[-3])
        assert base <= seed < base + per_task
        again = ub.generate(seed, inst.task, n_uav=inst.n_uav, n_users=len(inst.users),
                            scene=scene)
        again.save(tmp_path / "a.json")
        inst.save(tmp_path / "b.json")
        assert (tmp_path / "a.json").read_bytes() == (tmp_path / "b.json").read_bytes(), f.name
