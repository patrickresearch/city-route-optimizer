import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cityroute import heuristics, solve, viz
from cityroute.network import Network
from cityroute.scenario import city

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Heidelberg, Germany")
    p.add_argument("--customers", type=int, default=60)
    p.add_argument("--vehicles", type=int, default=8)
    p.add_argument("--capacity", type=int, default=60)
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    assets, results = ROOT / "assets", ROOT / "results"
    results.mkdir(exist_ok=True)
    net = Network(a.place)
    sc = city(net, a.customers, a.vehicles, a.capacity, a.seed)

    cvrp = [
        heuristics.nearest_neighbor(sc),
        heuristics.savings(sc),
        solve.ortools(sc, a.time_limit, False, "GREEDY_DESCENT", name="OR-Tools Local Search"),
        solve.ortools(sc, a.time_limit, False),
    ]
    vrptw = [
        solve.ortools(sc, a.time_limit, True, mh)
        for mh in ["GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH", "TABU_SEARCH"]
    ]
    table = pd.DataFrame(
        [
            dict(
                problem=prob,
                method=s.method,
                distance_km=round(s.distance, 2),
                vehicles=s.vehicles,
                runtime_s=round(s.runtime, 2),
            )
            for prob, runs in [("CVRP", cvrp), ("VRPTW", vrptw)]
            for s in runs
            if s
        ]
    )
    print(table.to_string(index=False))
    table.to_csv(results / "comparison.csv", index=False)

    viz.convergence([s for s in vrptw if s]).write_image(assets / "convergence.png", scale=2)

    rows = []
    for k in range(a.vehicles - 2, 2 * a.vehicles + 1):
        s = solve.ortools(replace(sc, vehicles=k), a.time_limit / 3, True, exact=True)
        if s:
            rows.append((k, s.distance))
            print(f"vehicles={k:2d} distance={s.distance:7.2f} km")
    viz.tradeoff(rows).write_image(assets / "tradeoff.png", scale=2)

    best = min((s for s in vrptw if s), key=lambda s: s.distance)
    viz.route_map(net, sc, best).save(results / "routes.html")
    viz.gif(net, sc, best, assets / "routes.gif")


if __name__ == "__main__":
    main()
