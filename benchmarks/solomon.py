import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import product
from pathlib import Path

import pandas as pd

from cityroute import solve, viz
from cityroute.scenario import solomon

HERE = Path(__file__).resolve().parent
SOLVERS = {
    "OR-Tools GLS": partial(solve.ortools, first="PARALLEL_CHEAPEST_INSERTION"),
    "PyVRP HGS": solve.hgs,
}


def run(job):
    path, method, limit = job
    sc = solomon(path)
    s = SOLVERS[method](sc, limit, True, fixed_cost=2 * sc.size * int(sc.dist.max()))
    return path.stem.upper(), method, s


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--pattern", default="*.txt")
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    bks = pd.read_csv(HERE / "bks.csv", index_col="instance")
    jobs = product(sorted((HERE / "solomon").glob(a.pattern)), SOLVERS, [a.time_limit])
    with ProcessPoolExecutor(a.workers) as pool:
        out = list(pool.map(run, jobs))
    table = pd.DataFrame(
        [
            dict(
                instance=name,
                group=name.rstrip("0123456789") + name[-3],
                method=method,
                distance=round(s.distance, 2),
                vehicles=s.vehicles,
                bks=bks.distance[name],
                bks_vehicles=bks.vehicles[name],
            )
            for name, method, s in out
            if s
        ]
    )
    table["extra_vehicles"] = table.vehicles - table.bks_vehicles
    table["gap_pct"] = 100 * (table.distance / table.bks - 1)
    (HERE.parent / "results").mkdir(exist_ok=True)
    table.to_csv(HERE.parent / "results" / "solomon.csv", index=False)

    same = table[table.extra_vehicles == 0]
    summary = (
        table.groupby(["group", "method"])
        .agg(instances=("instance", "size"), extra_vehicles=("extra_vehicles", "sum"))
        .join(same.groupby(["group", "method"]).gap_pct.agg(["mean", "max"]).round(2))
        .rename(columns={"mean": "gap_mean_pct", "max": "gap_max_pct"})
        .reset_index()
    )
    print(summary.to_string(index=False))
    viz.to_readme(summary, "solomon")


if __name__ == "__main__":
    main()
