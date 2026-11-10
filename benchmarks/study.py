import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon

from cityroute import heuristics, solve, viz
from cityroute.network import Network
from cityroute.scenario import city

ROOT = Path(__file__).resolve().parents[1]
SOLVERS = {
    "Solomon I1": lambda sc, limit: heuristics.insertion(sc),
    "OR-Tools GLS": lambda sc, limit: solve.ortools(sc, limit, True),
    "PyVRP HGS": lambda sc, limit: solve.hgs(sc, limit, True),
}


def run(job):
    customers, seed, method, sc, limit = job
    return customers, seed, method, SOLVERS[method](sc, limit)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Heidelberg, Germany")
    p.add_argument("--sizes", type=int, nargs="+", default=[40, 60, 100])
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--capacity", type=int, default=60)
    p.add_argument("--time-limit", type=float, default=10)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    net = Network(a.place)
    jobs = []
    for n, seed in product(a.sizes, range(a.seeds)):
        sc = replace(city(net, n, n // 4, a.capacity, seed), pred=None, net=None)
        jobs += [(n, seed, method, sc, a.time_limit) for method in SOLVERS]
    with ProcessPoolExecutor(a.workers) as pool:
        out = [r for r in pool.map(run, jobs) if r[3]]

    df = pd.DataFrame(
        [
            dict(customers=n, seed=s, method=m, distance=x.distance, vehicles=x.vehicles)
            for n, s, m, x in out
        ]
    )
    best = df.groupby(["customers", "seed"]).distance.transform("min")
    df["gap_pct"] = 100 * (df.distance / best - 1)
    df["integral_pct"] = [100 * x.primal_integral(b, a.time_limit) for (*_, x), b in zip(out, best)]
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "study.csv", index=False)

    summary = (
        df.groupby(["customers", "method"])
        .agg(
            gap_mean_pct=("gap_pct", "mean"),
            gap_max_pct=("gap_pct", "max"),
            integral_mean_pct=("integral_pct", "mean"),
            best_found=("gap_pct", lambda g: int((g == 0).sum())),
        )
        .round(2)
        .reset_index()
    )
    print(summary.to_string(index=False))
    viz.to_readme(summary, "study")

    wide = df.pivot_table(
        index=["customers", "seed"], columns="method", values=["gap_pct", "integral_pct"]
    )
    for metric in ["gap_pct", "integral_pct"]:
        test = wilcoxon(
            wide[metric]["PyVRP HGS"], wide[metric]["OR-Tools GLS"], zero_method="zsplit"
        )
        print(f"Wilcoxon HGS vs GLS, {metric}: p = {test.pvalue:.4f}")


if __name__ == "__main__":
    main()
