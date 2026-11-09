import argparse
from pathlib import Path

import pandas as pd

from cityroute import solve
from cityroute.scenario import solomon

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--time-limit", type=float, default=30)
    p.add_argument("--pattern", default="*.txt")
    a = p.parse_args()

    bks = pd.read_csv(HERE / "bks.csv", index_col="instance")
    rows = []
    for path in sorted((HERE / "solomon").glob(a.pattern)):
        s = solve.ortools(solomon(path), a.time_limit, True)
        name = path.stem.upper()
        ref = bks.distance.get(name)
        rows.append(
            dict(
                instance=name,
                distance=round(s.distance, 2),
                vehicles=s.vehicles,
                bks=ref,
                bks_vehicles=bks.vehicles.get(name),
                gap_pct=round(100 * (s.distance / ref - 1), 2) if ref else None,
            )
        )
        print(rows[-1])
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    (HERE.parent / "results").mkdir(exist_ok=True)
    table.to_csv(HERE.parent / "results" / "solomon.csv", index=False)


if __name__ == "__main__":
    main()
