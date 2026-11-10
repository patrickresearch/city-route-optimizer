from dataclasses import replace

import numpy as np

PROFILE = ((0, 1.4), (5400, 1.0), (25200, 1.3))


def travel(free, depart, profile=PROFILE):
    left, now = (x.astype(float) for x in np.broadcast_arrays(free, depart))
    ends = [start for start, _ in profile[1:]] + [np.inf]
    for (start, slow), end in zip(profile, ends):
        used = np.minimum(left, np.clip(end - np.maximum(now, start), 0, None) / slow)
        now, left = now + used * slow, left - used
    return now - depart


def bounded(sc, profile=PROFILE):
    lo, hi = (sc.windows + sc.service[:, None]).T
    hi = np.where(np.arange(sc.size) == 0, lo, hi)
    starts = [np.clip(start, lo, hi) for start, _ in profile]
    worst = np.max([travel(sc.time, d[:, None], profile) for d in [lo, hi, *starts]], axis=0)
    return replace(sc, time=np.ceil(worst).astype(int))


def lateness(sc, routes, profile=PROFILE):
    late = []
    for route in routes:
        t = float(sc.windows[0, 0])
        for a, b in zip([0, *route], [*route, 0]):
            t += sc.service[a]
            t = max(t + travel(sc.time[a, b], t, profile), sc.windows[b, 0])
            late.append(max(t - sc.windows[b, 1], 0.0))
    return np.array(late)
