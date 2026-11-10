import argparse
from itertools import product
from pathlib import Path

import pandas as pd

from cityroute import heuristics, solve, viz
from cityroute.network import Network
from cityroute.scenario import city

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Heidelberg, Germany")
    p.add_argument("--sizes", type=int, nargs="+", default=[20, 40, 60, 100])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--capacity", type=int, default=60)
    p.add_argument("--exact-limit", type=float, default=120)
    p.add_argument("--time-limit", type=float, default=5)
    a = p.parse_args()

    net = Network(a.place)
    rows = []
    for n, seed in product(a.sizes, range(a.seeds)):
        sc = city(net, n, n // 4, a.capacity, seed)
        runs = [
            heuristics.insertion(sc),
            solve.ortools(sc, a.time_limit),
            solve.hgs(sc, a.time_limit),
        ]
        exact = solve.cpsat(sc, a.exact_limit, hint=runs[-1].routes)
        runs.insert(0, exact)
        for s in runs:
            gap = 100 * (s.distance / exact.bound - 1)
            rows.append(
                dict(customers=n, seed=seed, method=s.method, distance=s.distance, gap_pct=gap)
            )
        print(f"n={n} seed={seed} best={exact.distance:.2f} bound={exact.bound:.2f}", end=" ")
        print(f"t={exact.runtime:.1f}s")

    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "exact.csv", index=False)
    summary = (
        df.groupby(["customers", "method"])
        .agg(
            gap_mean_pct=("gap_pct", "mean"),
            gap_max_pct=("gap_pct", "max"),
            optimal=("gap_pct", lambda g: int((g < 1e-6).sum())),
        )
        .round(2)
        .reset_index()
    )
    print(summary.to_string(index=False))
    viz.to_readme(summary, "exact")


if __name__ == "__main__":
    main()
