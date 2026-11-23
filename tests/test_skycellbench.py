import numpy as np
import pytest

import skycellbench as ub
from skycellbench.baselines import (centre_placement, kmeans_placement, local_search, probe_search,
                                random_placement)
from skycellbench.fast import _share, path_loss_db, spectral_efficiency


@pytest.fixture
def inst():
    return ub.generate(3, n_uav=4, n_users=80)


def sc(inst, u):
    return ub.score_fast(inst, u)["score"]


@pytest.mark.parametrize("task", ub.TASKS)
def test_generate_is_repeatable_and_round_trips(task, tmp_path):
    a = ub.generate(3, task, n_uav=4, n_users=60)
    assert ub.generate(3, task, n_uav=4, n_users=60) == a
    assert ub.generate(4, task, n_uav=4, n_users=60).users != a.users
    a.save(tmp_path / "i.json")
    assert ub.Instance.load(tmp_path / "i.json") == a
    p = ub.Placement(a.id, centre_placement(a).tolist())
    p.save(tmp_path / "p.json")
    assert ub.Placement.load(tmp_path / "p.json") == p


@pytest.mark.parametrize("bad", [
    [[0, 0, 100]] * 3,                       # wrong number of UAVs
    [[-1, 0, 100]] + [[0, 0, 100]] * 3,      # outside the area
    [[0, 0, 10]] + [[0, 0, 100]] * 3,        # below the altitude range
    [[0, 0, float("nan")]] + [[0, 0, 100]] * 3,
])
def test_invalid_placements_are_rejected(inst, bad):
    with pytest.raises(ValueError):
        ub.check(inst, bad)


def test_constrained_rules():
    inst = ub.generate(1, "constrained", n_uav=3, n_users=40)
    gx, gy = inst.gateway
    ok = np.tile([gx, gy, 60.0], (3, 1))
    ub.check(inst, ok)
    far = ok.copy()
    far[2, :2] = [abs(gx - 2000), abs(gy - 2000)]            # opposite corner: out of range
    with pytest.raises(ValueError, match="backhaul"):
        ub.check(inst, far)
    x, y, _ = inst.no_fly[0]
    with pytest.raises(ValueError, match="no-fly"):
        ub.check(inst, [[x, y, 60.0]] * 3)


def test_backhaul_may_be_relayed_through_another_uav():
    inst = ub.generate(1, "constrained", n_uav=2, n_users=40)
    inst.no_fly, inst.gateway, inst.backhaul_range_m = [], [0.0, 0.0], 500.0
    ub.check(inst, [[400, 0, 60], [800, 0, 60]])
    with pytest.raises(ValueError):
        ub.check(inst, [[800, 0, 60], [1200, 0, 60]])


def epochs_instance():
    """Three epochs made by hand; no task generates these at present."""
    inst = ub.generate(1, "place", n_uav=2, n_users=40)
    xy = [[x, y] for x, y, _ in inst.users]
    inst.tracks = [xy, xy, xy]
    inst.activity = [[1] * 40, [1] * 20 + [0] * 20, [1] * 40]
    inst.max_move_m = 100.0
    return inst


def test_rules_with_several_epochs():
    inst = epochs_instance()
    u = centre_placement(inst)
    assert u.shape == (3, 2, 3)
    ub.check(inst, u)
    with pytest.raises(ValueError):
        ub.check(inst, u[0])                                  # one epoch is not a plan
    u[1, 0, 0] += inst.max_move_m + 1
    with pytest.raises(ValueError, match="moves more"):
        ub.check(inst, u)


def test_score_with_several_epochs_is_the_mean_over_active_users():
    inst = epochs_instance()
    d = ub.score_fast(inst, centre_placement(inst), detail=True)
    n = [len(e["thr_mbps"]) for e in d["epochs"]]
    assert n == [40, 20, 40]
    assert np.isclose(d["score"], np.average([e["score"] for e in d["epochs"]], weights=n))


def test_blind_hides_users_and_counts_probes():
    inst = ub.generate(1, "blind", n_uav=3, n_users=60)
    pub = inst.public()
    assert pub.users == [] and inst.users and pub.n_uav == 3
    probe = ub.Probe(inst)
    u = probe_search(pub, probe, np.random.default_rng(0))
    assert probe.remaining == 0 and sc(inst, u) > sc(inst, centre_placement(inst))
    with pytest.raises(RuntimeError):
        probe(u)


@pytest.mark.parametrize("task", ub.TASKS)
def test_baselines_return_valid_placements_in_the_expected_order(task):
    inst = ub.generate(2, task, n_uav=4, n_users=80)
    rng = np.random.default_rng(0)
    for u in (centre_placement(inst), random_placement(inst, rng), kmeans_placement(inst, rng)):
        ub.check(inst, u)
    best = local_search(inst, rng, budget=600)
    ub.check(inst, best)
    assert sc(inst, centre_placement(inst)) < sc(inst, best)
    assert sc(inst, kmeans_placement(inst, rng)) <= sc(inst, best) + 0.05


def test_share_water_filling():
    assert np.allclose(_share(np.array([0.2, 0.3])), [0.2, 0.3])          # fits: all served
    got = _share(np.array([0.1, 0.8, 0.9]))
    assert np.allclose(got, [0.1, 0.45, 0.45]) and np.isclose(got.sum(), 1)


def test_path_loss_grows_with_distance_and_link_curve_is_monotone(inst):
    xy = np.array([[d, 0.0] for d in (0, 200, 800, 1600)])
    loss = path_loss_db(inst, np.array([[0.0, 0.0, 100.0]]), xy)[0]
    assert (np.diff(loss) > 0).all()
    se = spectral_efficiency(np.linspace(-10, 30, 200))
    assert se[0] == 0 and (np.diff(se) >= 0).all() and se[-1] > 3


def test_score_is_bounded_deterministic_and_ignores_uav_order(inst):
    rng = np.random.default_rng(0)
    for _ in range(20):
        u = random_placement(inst, rng)
        s = sc(inst, u)
        assert 0 <= s <= 1 and s == sc(inst, u) and np.isclose(s, sc(inst, u[:, ::-1]))


def test_more_demand_never_raises_the_score(inst):
    u = kmeans_placement(inst, np.random.default_rng(0))
    low = sc(inst, u)
    for x in inst.users:
        x[2] *= 3
    assert sc(inst, u) <= low + 1e-9


@pytest.mark.skipif(not ub.ns3.BINARY.exists(), reason="ns-3 scorer not built")
def test_ns3_scorer_agrees_on_a_small_case():
    inst = ub.generate(5, n_uav=2, n_users=20)
    u = kmeans_placement(inst, np.random.default_rng(0))
    assert abs(sc(inst, u) - ub.score_ns3(inst, u, sim_time=1.0)["score"]) < 0.1
