import argparse
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cityroute import heuristics, solve, viz
from cityroute.network import Network
from cityroute.scenario import city

ROOT = Path(__file__).resolve().parents[1]
SEARCH = ["GREEDY_DESCENT", "GUIDED_LOCAL_SEARCH", "TABU_SEARCH"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Heidelberg, Germany")
    p.add_argument("--customers", type=int, default=60)
    p.add_argument("--vehicles", type=int, default=8)
    p.add_argument("--capacity", type=int, default=60)
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--exact-limit", type=float, default=120)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    assets, results = ROOT / "assets", ROOT / "results"
    results.mkdir(exist_ok=True)
    net = Network(a.place)
    sc = city(net, a.customers, a.vehicles, a.capacity, a.seed)

    cvrp = [
        heuristics.nearest_neighbor(sc),
        heuristics.savings(sc),
        *(solve.ortools(sc, a.time_limit, False, mh) for mh in SEARCH[:2]),
        solve.hgs(sc, a.time_limit, False, a.seed),
    ]
    search = [solve.ortools(sc, a.time_limit, True, mh) for mh in SEARCH]
    search.append(solve.hgs(sc, a.time_limit, True, a.seed))
    vrptw = [heuristics.insertion(sc), *search]
    rows = []
    for prob, runs, windows in [("CVRP", cvrp, False), ("VRPTW", vrptw, True)]:
        best = min((s for s in runs[-2:] if s), key=lambda s: s.distance)
        bound = solve.cpsat(sc, a.exact_limit, windows, hint=best.routes).bound
        rows += [
            dict(
                problem=prob,
                method=s.method,
                distance_km=round(s.distance, 2),
                gap_pct=round(100 * (s.distance / bound - 1), 2),
                vehicles=s.vehicles,
                runtime_s=round(s.runtime, 2),
            )
            for s in runs
            if s
        ]
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(results / "comparison.csv", index=False)
    viz.to_readme(table, "comparison")

    viz.convergence([s for s in search if s]).write_image(assets / "convergence.png", scale=2)

    curve = []
    for k in range(-(-sc.demand.sum() // a.capacity), 2 * a.vehicles + 1):
        s = solve.ortools(replace(sc, vehicles=k), a.time_limit / 3, True, use_all=True)
        print(
            f"vehicles={k:2d} "
            + (f"distance={s.distance:7.2f} km used={s.vehicles}" if s else "no solution")
        )
        if s and s.vehicles == k:
            curve.append((k, s.distance))
    viz.tradeoff(curve).write_image(assets / "tradeoff.png", scale=2)

    best = min((s for s in search if s), key=lambda s: s.distance)
    viz.route_map(sc, best).save(results / "routes.html")
    viz.gif(sc, best, assets / "routes.gif")


if __name__ == "__main__":
    main()
