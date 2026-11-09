from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Scenario:
    dist: np.ndarray
    time: np.ndarray
    demand: np.ndarray
    windows: np.ndarray
    service: np.ndarray
    vehicles: int
    capacity: int
    scale: float = 1.0
    sites: np.ndarray | None = None
    pred: np.ndarray | None = None

    @property
    def size(self):
        return len(self.demand)


def city(net, customers=40, vehicles=6, capacity=60, seed=0, horizon=8 * 3600, service=300):
    rng = np.random.default_rng(seed)
    depot = net.nearest(np.median(net.xy, axis=0))
    snapped = net.nearest(rng.permutation(net.buildings()))
    _, first = np.unique(snapped, return_index=True)
    picks = snapped[np.sort(first)]
    sites = np.r_[depot, picks[picks != depot][:customers]]
    dist, time, pred = net.matrices(sites)
    width = rng.choice([3600, 7200], size=customers)
    start = rng.integers(0, horizon - width)
    return Scenario(
        dist=np.rint(dist).astype(int),
        time=np.ceil(time).astype(int),
        demand=np.r_[0, rng.integers(1, 10, customers)],
        windows=np.vstack([[0, horizon + 3600], np.column_stack([start, start + width])]),
        service=np.r_[0, np.full(customers, service)],
        vehicles=vehicles,
        capacity=capacity,
        scale=1000.0,
        sites=sites,
        pred=pred,
    )


def solomon(path, vehicles=None, scale=100):
    rows = [r.split() for r in Path(path).read_text().splitlines()]
    numeric = [r for r in rows if r and all(t.replace(".", "", 1).isdigit() for t in r)]
    nums = [list(map(float, r)) for r in numeric]
    fleet, capacity = map(int, nums[0])
    data = np.array(nums[1:])
    xy = data[:, 1:3]
    dist = np.rint(np.linalg.norm(xy[:, None] - xy[None], axis=2) * scale).astype(int)
    return Scenario(
        dist=dist,
        time=dist,
        demand=data[:, 3].astype(int),
        windows=np.rint(data[:, 4:6] * scale).astype(int),
        service=np.rint(data[:, 6] * scale).astype(int),
        vehicles=vehicles or fleet,
        capacity=capacity,
        scale=scale,
    )
