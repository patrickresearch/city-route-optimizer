import time as clock
from dataclasses import dataclass, field
from itertools import accumulate

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from ortools.sat.python import cp_model
from pyvrp import Model
from pyvrp.stop import MaxRuntime


@dataclass
class Solution:
    method: str
    routes: list
    distance: float
    runtime: float
    trace: list = field(default_factory=list)
    bound: float | None = None

    @property
    def vehicles(self):
        return len(self.routes)

    def primal_integral(self, best, horizon):
        trace = self.trace or [(self.runtime, self.distance)]
        times = [min(t, horizon) for t, _ in trace] + [horizon]
        gaps = [1.0] + [(c - best) / c for _, c in trace]
        return sum(g * (b - a) for g, a, b in zip(gaps, [0.0, *times[:-1]], times)) / horizon


def distance(sc, routes):
    return sum(sc.dist[[0, *r], [*r, 0]].sum() for r in routes) / sc.scale


def _improvements(trace, runtime, total):
    out = []
    for t, c in trace:
        if not out or c < out[-1][1]:
            out.append((t, c))
    return [*out, (runtime, total)]


def ortools(
    sc,
    time_limit=10.0,
    windows=True,
    metaheuristic="GUIDED_LOCAL_SEARCH",
    first="PATH_CHEAPEST_ARC",
    use_all=False,
    fixed_cost=0,
):
    n, k = sc.size, sc.vehicles
    manager = pywrapcp.RoutingIndexManager(n, k, 0)
    model = pywrapcp.RoutingModel(manager)
    model.SetArcCostEvaluatorOfAllVehicles(model.RegisterTransitMatrix(sc.dist.tolist()))
    model.SetFixedCostOfAllVehicles(int(fixed_cost))
    load = model.RegisterUnaryTransitVector(sc.demand.tolist())
    model.AddDimensionWithVehicleCapacity(load, 0, [sc.capacity] * k, True, "load")
    if windows:
        travel = model.RegisterTransitMatrix((sc.time + sc.service[:, None]).tolist())
        horizon = int(sc.windows[0, 1])
        model.AddDimension(travel, horizon, horizon, False, "time")
        dim = model.GetDimensionOrDie("time")
        for node in range(1, n):
            dim.CumulVar(manager.NodeToIndex(node)).SetRange(*sc.windows[node].tolist())
        for v in range(k):
            dim.CumulVar(model.Start(v)).SetRange(*sc.windows[0].tolist())
            dim.CumulVar(model.End(v)).SetRange(*sc.windows[0].tolist())
    if use_all:
        model.AddConstantDimension(1, n, True, "visits")
        visits = model.GetDimensionOrDie("visits")
        for v in range(k):
            visits.SetCumulVarSoftLowerBound(model.End(v), 2, 10**7)
            model.SetVehicleUsedWhenEmpty(True, v)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = getattr(routing_enums_pb2.FirstSolutionStrategy, first)
    params.local_search_metaheuristic = getattr(
        routing_enums_pb2.LocalSearchMetaheuristic, metaheuristic
    )
    params.time_limit.FromMilliseconds(int(time_limit * 1000))

    trace = []
    start = clock.perf_counter()
    model.AddAtSolutionCallback(
        lambda: trace.append((clock.perf_counter() - start, model.CostVar().Value() / sc.scale))
    )
    solution = model.SolveWithParameters(params)
    runtime = clock.perf_counter() - start
    if solution is None:
        return None

    routes = []
    for v in range(k):
        i, route = solution.Value(model.NextVar(model.Start(v))), []
        while not model.IsEnd(i):
            route.append(manager.IndexToNode(i))
            i = solution.Value(model.NextVar(i))
        if route:
            routes.append(route)
    total = distance(sc, routes)
    label = f"OR-Tools {metaheuristic.replace('_', ' ').title()}"
    return Solution(label, routes, total, runtime, _improvements(trace, runtime, total))


def hgs(sc, time_limit=10.0, windows=True, seed=0, fixed_cost=0):
    n = sc.size
    tw = sc.windows if windows else np.tile([0, 2**40], (n, 1))
    model = Model()
    locs = [model.add_location(0, 0) for _ in range(n)]
    model.add_depot(locs[0])
    model.add_vehicle_type(
        sc.vehicles,
        int(sc.capacity),
        fixed_cost=int(fixed_cost),
        tw_early=int(tw[0, 0]),
        tw_late=int(tw[0, 1]),
    )
    for i in range(1, n):
        model.add_client(
            locs[i],
            delivery=int(sc.demand[i]),
            service_duration=int(sc.service[i]),
            tw_early=int(tw[i, 0]),
            tw_late=int(tw[i, 1]),
        )
    for i, a in enumerate(locs):
        for j, b in enumerate(locs):
            model.add_edge(a, b, distance=int(sc.dist[i, j]), duration=int(sc.time[i, j]))

    start = clock.perf_counter()
    result = model.solve(MaxRuntime(time_limit), seed=seed, display=False)
    runtime = clock.perf_counter() - start
    if not result.is_feasible():
        return None

    routes = [[a.idx + 1 for a in r if a.is_client()] for r in result.best.routes()]
    total = distance(sc, routes)
    stats = zip(accumulate(result.stats.runtimes), result.stats.data)
    trace = [(t, d.best_cost / sc.scale) for t, d in stats if d.best_feas]
    return Solution("PyVRP HGS", routes, total, runtime, _improvements(trace, runtime, total))


def cpsat(sc, time_limit=60.0, windows=True, workers=8, hint=None):
    n = sc.size
    m = cp_model.CpModel()
    x = {(i, j): m.new_bool_var("") for i in range(n) for j in range(n) if i != j}
    m.add_multiple_circuit([(i, j, v) for (i, j), v in x.items()])
    m.add(sum(x[0, j] for j in range(1, n)) <= sc.vehicles)
    load = [m.new_int_var(int(sc.demand[i]), int(sc.capacity), "") for i in range(n)]
    begin = [m.new_int_var(int(sc.windows[i, 0]), int(sc.windows[i, 1]), "") for i in range(n)]
    for (i, j), v in x.items():
        if j:
            m.add(load[j] >= load[i] + int(sc.demand[j])).only_enforce_if(v)
            if windows:
                travel = int(sc.service[i] + sc.time[i, j])
                m.add(begin[j] >= begin[i] + travel).only_enforce_if(v)
        elif windows:
            back = int(sc.service[i] + sc.time[i, 0])
            m.add(begin[i] + back <= int(sc.windows[0, 1])).only_enforce_if(v)
    m.minimize(cp_model.LinearExpr.weighted_sum(list(x.values()), [int(sc.dist[k]) for k in x]))
    if hint:
        arcs = {arc for r in hint for arc in zip([0, *r], [*r, 0])}
        for arc, v in x.items():
            m.add_hint(v, arc in arcs)
        for r in hint:
            q, t = 0, int(sc.windows[0, 0])
            for a, b in zip([0, *r[:-1]], r):
                q += int(sc.demand[b])
                t = max(int(sc.windows[b, 0]), t + int(sc.service[a] + sc.time[a, b]))
                m.add_hint(load[b], q)
                m.add_hint(begin[b], t)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit
    solver.parameters.num_workers = workers
    start = clock.perf_counter()
    status = solver.solve(m)
    runtime = clock.perf_counter() - start
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None

    succ = {i: j for (i, j), v in x.items() if i and solver.value(v)}
    routes = []
    for j in range(1, n):
        if solver.value(x[0, j]):
            routes.append([j])
            while succ[routes[-1][-1]]:
                routes[-1].append(succ[routes[-1][-1]])
    total = distance(sc, routes)
    return Solution("CP-SAT", routes, total, runtime, bound=solver.best_objective_bound / sc.scale)
