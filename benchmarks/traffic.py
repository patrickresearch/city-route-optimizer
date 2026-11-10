import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import product
from pathlib import Path

import pandas as pd

from cityroute import robust, solve, traffic, viz
from cityroute.network import Network
from cityroute.scenario import city

ROOT = Path(__file__).resolve().parents[1]
PLANS = {
    "Free-flow times": lambda sc: sc,
    "Peak factor everywhere": lambda sc: robust.buffered(
        sc, max(f for _, f in traffic.PROFILE) - 1
    ),
    "Worst case per departure window": traffic.bounded,
}


def run(job):
    seed, plan, sc, limit = job
    s = solve.hgs(PLANS[plan](sc), limit, seed=seed)
    if s is None:
        return None
    late = traffic.lateness(sc, s.routes)
    return dict(
        seed=seed,
        plan=plan,
        distance=s.distance,
        vehicles=s.vehicles,
        late_stops=(late > 0).mean(),
        max_late_min=late.max() / 60,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--place", default="Heidelberg, Germany")
    p.add_argument("--customers", type=int, default=60)
    p.add_argument("--vehicles", type=int, default=10)
    p.add_argument("--capacity", type=int, default=60)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--time-limit", type=float, default=10)
    p.add_argument("--workers", type=int, default=4)
    a = p.parse_args()

    net = Network(a.place)
    jobs = [
        (
            seed,
            plan,
            replace(city(net, a.customers, a.vehicles, a.capacity, seed), pred=None, net=None),
            a.time_limit,
        )
        for seed, plan in product(range(a.seeds), PLANS)
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        df = pd.DataFrame([r for r in pool.map(run, jobs) if r])
    base = df.seed.map(df[df.plan == "Free-flow times"].set_index("seed").distance)
    df["extra_km_pct"] = 100 * (df.distance / base - 1)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results" / "traffic.csv", index=False)

    summary = df.groupby("plan", sort=False).agg(
        instances=("seed", "size"),
        extra_km_pct=("extra_km_pct", "mean"),
        vehicles=("vehicles", "mean"),
        plans_with_delay=("late_stops", lambda x: int((x > 0).sum())),
        late_stops_pct=("late_stops", lambda x: 100 * x.mean()),
        max_late_min=("max_late_min", "max"),
    )
    summary = summary.round(2).reset_index()
    print(summary.to_string(index=False))
    viz.to_readme(summary, "traffic")


if __name__ == "__main__":
    main()
