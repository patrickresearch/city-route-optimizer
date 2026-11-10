import time as clock

import numpy as np

from cityroute.solve import Solution, distance

LAMBDAS = np.round(np.arange(0.4, 1.65, 0.1), 1)
I1_GRID = [(1.0, lam, a1) for lam in (1.0, 2.0) for a1 in (0.0, 0.5, 1.0)]


def nearest_neighbor(sc):
    start = clock.perf_counter()
    left = np.ones(sc.size, bool)
    left[0] = False
    routes = []
    while left.any():
        route, load, cur = [], 0, 0
        while (ok := left & (load + sc.demand <= sc.capacity)).any():
            cur = int(np.argmin(np.where(ok, sc.dist[cur], np.inf)))
            route.append(cur)
            load += sc.demand[cur]
            left[cur] = False
        routes.append(route)
    return Solution("Nearest Neighbor", routes, distance(sc, routes), clock.perf_counter() - start)


def _savings(sc, lam):
    d, n = sc.dist, sc.size
    s = d[1:, :1] + d[:1, 1:] - lam * d[1:, 1:]
    owner = np.arange(n)
    routes = {i: [i] for i in range(1, n)}
    load = sc.demand.copy()
    for flat in np.argsort(-s, axis=None):
        i, j = divmod(int(flat), n - 1)
        if s[i, j] <= 0:
            break
        a, b = owner[i + 1], owner[j + 1]
        if a == b or routes[a][-1] != i + 1 or routes[b][0] != j + 1:
            continue
        if load[a] + load[b] > sc.capacity:
            continue
        routes[a] += routes.pop(b)
        load[a] += load[b]
        owner[routes[a]] = a
    return list(routes.values())


def savings(sc, lambdas=LAMBDAS):
    start = clock.perf_counter()
    routes = min((_savings(sc, lam) for lam in lambdas), key=lambda r: distance(sc, r))
    runtime = clock.perf_counter() - start
    return Solution("Clarke-Wright Savings", routes, distance(sc, routes), runtime)


def _schedule(sc, path):
    begin, latest = np.empty(len(path)), np.empty(len(path))
    begin[0], latest[-1] = sc.windows[0, 0], sc.windows[0, 1]
    for k in range(1, len(path)):
        a, v = path[k - 1], path[k]
        begin[k] = max(sc.windows[v, 0], begin[k - 1] + sc.service[a] + sc.time[a, v])
    for k in range(len(path) - 2, -1, -1):
        v, w = path[k], path[k + 1]
        latest[k] = min(sc.windows[v, 1], latest[k + 1] - sc.service[v] - sc.time[v, w])
    return begin, latest


def _i1(sc, mu, lam, a1):
    d, t, s, q = sc.dist, sc.time, sc.service, sc.demand
    lo, hi = sc.windows.T
    left = np.ones(sc.size, bool)
    left[0] = False
    routes = []
    while left.any():
        free = np.flatnonzero(left)
        route = [free[np.argmax(d[0, free])]]
        left[route[0]] = False
        while (u := np.flatnonzero(left & (q[route].sum() + q <= sc.capacity))).size:
            path = np.array([0, *route, 0])
            b, z = _schedule(sc, path)
            i, j = path[:-1], path[1:]
            bu = np.maximum(lo[u, None], b[:-1] + s[i] + t[i][:, u].T)
            bj = np.maximum(lo[j], bu + s[u, None] + t[u][:, j])
            c1 = a1 * (d[i][:, u].T + d[u][:, j] - mu * d[i, j]) + (1 - a1) * (bj - b[1:])
            c1[(bu > hi[u, None]) | (bj > z[1:])] = np.inf
            pos = c1.argmin(1)
            cost = c1[np.arange(u.size), pos]
            if np.isinf(cost).all():
                break
            k = np.argmax(np.where(np.isinf(cost), -np.inf, lam * d[0, u] - cost))
            route.insert(pos[k], u[k])
            left[u[k]] = False
        routes.append(route)
    return routes


def insertion(sc, grid=I1_GRID):
    start = clock.perf_counter()
    routes = min((_i1(sc, *g) for g in grid), key=lambda r: (len(r) > sc.vehicles, distance(sc, r)))
    runtime = clock.perf_counter() - start
    return Solution("Solomon I1", routes, distance(sc, routes), runtime)
