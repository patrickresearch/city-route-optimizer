import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from cityroute import robust, solve, viz
from cityroute.network import Network
from cityroute.scenario import city

ROOT = Path(__file__).resolve().parents[1]


def run(job):
    seed, buffer, sc, limit, cv, shared = job
    s = solve.hgs(robust.buffered(sc, buffer), limit, seed=seed)
    if s is None:
        return None
    sim = robust.simulate(sc, s.routes, cv, shared, seed=seed)
    return dict(
        seed=seed, buffer_pct=round(100 * buffer), distance=s.distance, vehicles=s.vehicles, **sim
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Heidelberg, Germany")
    p.add_argument("--customers", type=int, default=60)
    p.add_argument("--vehicles", type=int, default=10)
    p.add_argument("--capacity", type=int, default=60)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument(
        "--buffers", type=float, nargs="+", default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1]
    )
    p.add_argument("--cv", type=float, default=0.3)
    p.add_argument("--shared", type=float, default=0.5)
    p.add_argument("--time-limit", type=float, default=10)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    net = Network(a.place)
    scenarios = {
        seed: replace(city(net, a.customers, a.vehicles, a.capacity, seed), pred=None, net=None)
        for seed in range(a.seeds)
    }
    jobs = [
        (seed, b, scenarios[seed], a.time_limit, a.cv, a.shared)
        for seed, b in product(scenarios, a.buffers)
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        df = pd.DataFrame([r for r in pool.map(run, jobs) if r])
    base = df.seed.map(df[df.buffer_pct == 0].set_index("seed").distance)
    df["extra_km_pct"] = 100 * (df.distance / base - 1)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "robust.csv", index=False)

    summary = df.groupby("buffer_pct").agg(
        instances=("seed", "size"),
        extra_km_pct=("extra_km_pct", "mean"),
        vehicles=("vehicles", "mean"),
        stops_on_time_pct=("stops_on_time", lambda x: 100 * x.mean()),
        days_on_time_pct=("days_on_time", lambda x: 100 * x.mean()),
        late_min=("late_min", "mean"),
    )
    summary = summary.round(1).reset_index()
    print(summary.to_string(index=False))
    viz.to_readme(summary, "robust")
    viz.robustness(summary).write_image(ROOT / "assets" / "robustness.png", scale=2)


if __name__ == "__main__":
    main()
