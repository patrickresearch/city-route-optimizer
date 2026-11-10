import numpy as np
import pytest
from scipy.sparse import csr_matrix

from cityroute import heuristics, robust, solve, traffic
from cityroute.network import attach, shortest_paths
from cityroute.scenario import Scenario


def toy(n=30, vehicles=6, capacity=40, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * 1000
    dist = np.rint(np.linalg.norm(xy[:, None] - xy[None], axis=2)).astype(int)
    start = rng.integers(0, 3000, n)
    windows = np.column_stack([start, start + 1500])
    windows[0] = [0, 10_000]
    return Scenario(
        dist=dist,
        time=dist,
        demand=np.r_[0, rng.integers(1, 10, n - 1)],
        windows=windows,
        service=np.r_[0, np.full(n - 1, 50)],
        vehicles=vehicles,
        capacity=capacity,
    )


def assert_feasible(sc, sol, windows=False):
    assert sorted(c for r in sol.routes for c in r) == list(range(1, sc.size))
    assert all(sc.demand[r].sum() <= sc.capacity for r in sol.routes)
    assert sol.distance == pytest.approx(solve.distance(sc, sol.routes))
    for route in sol.routes if windows else []:
        t = sc.windows[0, 0]
        for a, b in zip([0, *route], [*route, 0]):
            t = max(t + sc.service[a] + sc.time[a, b], sc.windows[b, 0])
            assert t <= sc.windows[b, 1]


@pytest.mark.parametrize("method", [heuristics.nearest_neighbor, heuristics.savings])
def test_heuristics_feasible(method):
    sc = toy()
    assert_feasible(sc, method(sc))


def test_insertion_respects_windows():
    sc = toy(vehicles=29)
    assert_feasible(sc, heuristics.insertion(sc), windows=True)


@pytest.mark.parametrize("solver", [solve.ortools, solve.hgs])
def test_search_beats_savings(solver):
    sc = toy()
    best = solver(sc, 2, windows=False)
    assert_feasible(sc, best)
    assert best.distance <= heuristics.savings(sc).distance + 1e-9


@pytest.mark.parametrize("solver", [solve.ortools, solve.hgs])
def test_search_respects_windows(solver):
    sc = toy(vehicles=29)
    assert_feasible(sc, solver(sc, 2, windows=True), windows=True)


def test_cpsat_proves_optimum():
    sc = toy(n=9, vehicles=8)
    opt = solve.cpsat(sc, 20)
    assert_feasible(sc, opt, windows=True)
    assert opt.distance == pytest.approx(opt.bound)
    assert opt.distance <= solve.hgs(sc, 1).distance + 1e-9


def test_primal_integral():
    sol = solve.Solution("x", [], 10.0, 2.0, trace=[(1.0, 20.0), (2.0, 10.0)])
    assert sol.primal_integral(10.0, 4.0) == pytest.approx((1.0 + 0.5) / 4.0)


def test_path_lengths_follow_fastest_paths():
    rng = np.random.default_rng(0)
    n, m = 150, 1500
    r, c = rng.integers(0, n, (2, m))
    time = csr_matrix((rng.random(m) + 0.1, (r, c)), shape=(n, n))
    length = csr_matrix((rng.random(m), (r, c)), shape=(n, n))
    sites = rng.choice(n, 8, replace=False)
    dist, t, pred = shortest_paths(time, length, sites)
    for a, src in enumerate(sites):
        for b, dst in enumerate(sites):
            if np.isinf(t[a, b]):
                continue
            total, v = 0.0, dst
            while v != src:
                total += length[pred[a, v], v]
                v = pred[a, v]
            assert dist[a, b] == pytest.approx(total)


def test_simulation_matches_plan_without_noise():
    sc = toy(vehicles=29)
    sol = solve.hgs(sc, 1)
    assert robust.simulate(sc, sol.routes, cv=0.0, reps=10)["days_on_time"] == 1.0


def test_buffer_buys_punctuality():
    sc = toy(vehicles=29)
    plain = robust.simulate(sc, solve.hgs(sc, 1).routes, cv=0.3)
    padded = robust.simulate(sc, solve.hgs(robust.buffered(sc, 0.3), 1).routes, cv=0.3)
    assert plain["stops_on_time"] < padded["stops_on_time"] <= 1.0
    assert plain["days_on_time"] < padded["days_on_time"]


def test_cpsat_keeps_feasible_hint():
    sc = toy(n=25, vehicles=24)
    start = solve.hgs(sc, 1)
    opt = solve.cpsat(sc, 0.5, hint=start.routes)
    assert opt.bound - 1e-9 <= opt.distance <= start.distance + 1e-9


def test_stops_sit_on_their_street():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]])
    pairs = [(0, 1), (1, 0), (1, 2), (2, 3), (3, 0)]
    r, c = zip(*pairs)
    length = csr_matrix((np.full(5, 10.0), (r, c)), shape=(4, 4))
    edges = {k: xy[list(k)] for k in pairs}
    stops = [[2.0, 1.0], [7.0, -1.0], [4.0, 11.0], [8.0, 11.0]]
    xy2, length2, time2, edges2, sites = attach(xy, length, 2 * length, edges, np.ones(2), stops)
    assert xy2[sites] == pytest.approx(np.array([[2, 0], [7, 0], [4, 10], [8, 10]]))
    dist, time, pred = shortest_paths(time2, length2, sites)
    assert dist == pytest.approx(
        np.array([[0, 5, 24, 20], [5, 0, 19, 15], [16, 21, 0, 36], [20, 25, 4, 0]])
    )
    assert time == pytest.approx(2 * dist)
    assert edges2[sites[0], sites[1]] == pytest.approx(np.array([[2, 0], [7, 0]]))


def test_travel_time_is_fifo_and_exact():
    assert traffic.travel(1000, 5000) == pytest.approx(400 + 1000 - 400 / 1.4)
    rng = np.random.default_rng(0)
    free, depart = rng.uniform(0, 4000, 500), np.sort(rng.uniform(0, 30_000, 500))
    arrive = depart + traffic.travel(free[:1], depart)
    assert (np.diff(arrive) >= -1e-9).all()
    assert (traffic.travel(free, depart) >= free - 1e-9).all()


def test_bounded_plan_is_punctual_in_traffic():
    sc = toy(vehicles=29)
    profile = ((0, 1.5), (1500, 1.0), (3000, 1.3))
    plain = traffic.lateness(sc, solve.hgs(sc, 1).routes, profile)
    safe = traffic.lateness(sc, solve.hgs(traffic.bounded(sc, profile), 1).routes, profile)
    assert plain.max() > 0
    assert safe.max() == 0
