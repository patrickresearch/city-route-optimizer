import numpy as np
import pytest

from cityroute import heuristics, solve
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


def assert_feasible(sc, sol):
    assert sorted(c for r in sol.routes for c in r) == list(range(1, sc.size))
    assert all(sc.demand[r].sum() <= sc.capacity for r in sol.routes)
    assert sol.distance == pytest.approx(solve.distance(sc, sol.routes))


@pytest.mark.parametrize("method", [heuristics.nearest_neighbor, heuristics.savings])
def test_heuristics_feasible(method):
    sc = toy()
    assert_feasible(sc, method(sc))


def test_ortools_beats_heuristics():
    sc = toy()
    best = solve.ortools(sc, 2, windows=False)
    assert_feasible(sc, best)
    assert best.distance <= heuristics.savings(sc).distance + 1e-9


def test_ortools_respects_windows():
    sc = toy(vehicles=29)
    sol = solve.ortools(sc, 2, windows=True)
    assert_feasible(sc, sol)
    for route in sol.routes:
        t = 0
        for a, b in zip([0, *route], route):
            t = max(t + sc.service[a] + sc.time[a, b], sc.windows[b, 0])
            assert t <= sc.windows[b, 1]
