import time as clock

import numpy as np

from cityroute.solve import Solution, distance


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


def savings(sc):
    start = clock.perf_counter()
    d, n = sc.dist, sc.size
    s = d[1:, :1] + d[:1, 1:] - d[1:, 1:]
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
    result = list(routes.values())
    runtime = clock.perf_counter() - start
    return Solution("Clarke-Wright Savings", result, distance(sc, result), runtime)
