import time as clock
from dataclasses import dataclass, field
from itertools import accumulate

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


@dataclass
class Solution:
    method: str
    routes: list
    distance: float
    runtime: float
    trace: list = field(default_factory=list)

    @property
    def vehicles(self):
        return len(self.routes)


def distance(sc, routes):
    return sum(sc.dist[[0, *r], [*r, 0]].sum() for r in routes) / sc.scale


def ortools(
    sc,
    time_limit=10.0,
    windows=True,
    metaheuristic="GUIDED_LOCAL_SEARCH",
    first="PATH_CHEAPEST_ARC",
    name=None,
    exact=False,
):
    n, k = sc.size, sc.vehicles
    manager = pywrapcp.RoutingIndexManager(n, k, 0)
    model = pywrapcp.RoutingModel(manager)
    model.SetArcCostEvaluatorOfAllVehicles(model.RegisterTransitMatrix(sc.dist.tolist()))
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
    if exact:
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
    label = name or f"OR-Tools {metaheuristic.replace('_', ' ').title()}"
    trace.append((runtime, total))
    best = accumulate((c for _, c in trace), min)
    return Solution(label, routes, total, runtime, [(t, c) for (t, _), c in zip(trace, best)])
