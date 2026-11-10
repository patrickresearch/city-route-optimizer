from dataclasses import replace

import numpy as np


def buffered(sc, buffer):
    return replace(sc, time=np.ceil(sc.time * (1 + buffer)).astype(int))


def simulate(sc, routes, cv=0.3, shared=0.5, reps=2000, seed=0):
    rng = np.random.default_rng(seed)
    sigma = np.sqrt(np.log1p(cv**2))
    day = np.sqrt(shared) * rng.standard_normal((reps, 1))
    late = []
    for route in routes:
        a, b = np.array([0, *route[:-1]]), np.array(route)
        z = day + np.sqrt(1 - shared) * rng.standard_normal((reps, len(route)))
        travel = sc.time[a, b] * np.exp(sigma * z - sigma**2 / 2)
        t = np.full(reps, float(sc.windows[0, 0]))
        for k in range(len(route)):
            t = np.maximum(t + sc.service[a[k]] + travel[:, k], sc.windows[b[k], 0])
            late.append(np.maximum(t - sc.windows[b[k], 1], 0))
    late = np.column_stack(late)
    return dict(
        stops_on_time=(late == 0).mean(),
        days_on_time=(late == 0).all(1).mean(),
        late_min=late[late > 0].mean() / 60 if late.any() else 0.0,
    )
