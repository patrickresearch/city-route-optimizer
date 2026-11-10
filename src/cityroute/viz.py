import re
from datetime import UTC, datetime
from pathlib import Path

import folium
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import plotly.graph_objects as go
from folium.plugins import TimestampedGeoJson
from matplotlib.animation import FuncAnimation, PillowWriter

COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
    "#bcbd22",
    "#7f7f7f",
]
SHIFT_START = datetime(2026, 1, 1, 8, tzinfo=UTC)
README = Path(__file__).resolve().parents[2] / "README.md"


def trajectories(sc, sol, windows=True, tolerance=1e-4):
    net = sc.net
    out = []
    for route in sol.routes:
        stops, t, coords, times = [0, *route, 0], 0.0, [], []
        for a, b in zip(stops[:-1], stops[1:]):
            line = net.line(net.path(sc.pred, sc.sites, a, b), tolerance)
            along = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
            coords.append(line)
            times.append(t + sc.time[a, b] * along / along[-1])
            t += sc.time[a, b]
            if windows:
                t = max(t, sc.windows[b, 0])
            t += sc.service[b]
        out.append((np.vstack(coords), np.concatenate(times)))
    return out


def _clock(seconds):
    m = int(seconds) // 60 + SHIFT_START.hour * 60
    return f"{m // 60:02d}:{m % 60:02d}"


def route_map(sc, sol, windows=True):
    pts = sc.net.xy[sc.sites]
    m = folium.Map(pts.mean(0)[::-1].tolist(), zoom_start=13)
    epoch = SHIFT_START.timestamp() * 1000
    features = []
    for k, (coords, times) in enumerate(trajectories(sc, sol, windows)):
        color = COLORS[k % len(COLORS)]
        folium.PolyLine(coords[:, ::-1].tolist(), color=color, weight=4, opacity=0.8).add_to(m)
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords.tolist()},
                "properties": {
                    "times": (epoch + times * 1000).astype(int).tolist(),
                    "style": {"color": color, "weight": 5},
                },
            }
        )
    for i, (x, y) in enumerate(pts):
        start, end = (_clock(s) for s in sc.windows[i])
        tip = "Depot" if i == 0 else f"#{i} · demand {sc.demand[i]} · {start}–{end}"
        folium.CircleMarker(
            (y, x),
            radius=7 if i == 0 else 4,
            color="black",
            fill=True,
            fill_color="black" if i == 0 else "white",
            fill_opacity=1,
            tooltip=tip,
        ).add_to(m)
    TimestampedGeoJson(
        {"type": "FeatureCollection", "features": features},
        period="PT1M",
        add_last_point=True,
        transition_time=40,
        loop=False,
    ).add_to(m)
    return m


def gif(sc, sol, path, windows=True, frames=150, fps=25):
    traj = trajectories(sc, sol, windows)
    pts = sc.net.xy[sc.sites]
    lo, hi = pts.min(0), pts.max(0)
    pad = 0.06 * (hi - lo)
    lo, hi = lo - pad, hi + pad
    w, h = (hi - lo) * [np.cos(np.radians(pts[:, 1].mean())), 1]
    fig, ax = ox.plot_graph(
        sc.net.graph,
        show=False,
        close=False,
        node_size=0,
        edge_color="#d9d9d9",
        edge_linewidth=0.5,
        bgcolor="white",
        figsize=(8, 8 * h / w),
    )
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.scatter(*pts[1:].T, s=12, c="white", edgecolors="black", linewidths=0.6, zorder=3)
    ax.scatter(*pts[0], s=90, c="black", marker="s", zorder=4)
    colors = [COLORS[k % len(COLORS)] for k in range(len(traj))]
    trails = [ax.plot([], [], color=c, lw=2, zorder=2)[0] for c in colors]
    dots = [ax.plot([], [], "o", color=c, ms=7, mec="black", zorder=5)[0] for c in colors]
    label = ax.text(0.02, 0.97, "", transform=ax.transAxes, fontsize=14, va="top")
    end = max(t[-1] for _, t in traj)

    def draw(f):
        now = end * f / (frames - 1)
        for (c, t), trail, dot in zip(traj, trails, dots):
            x, y = np.interp(now, t, c[:, 0]), np.interp(now, t, c[:, 1])
            seen = t <= now
            trail.set_data(np.r_[c[seen, 0], x], np.r_[c[seen, 1], y])
            dot.set_data([x], [y])
        label.set_text(_clock(now))
        return [*trails, *dots, label]

    FuncAnimation(fig, draw, frames=frames, blit=True).save(
        path, writer=PillowWriter(fps=fps), dpi=90
    )
    plt.close(fig)


def convergence(solutions):
    fig = go.Figure()
    for s in solutions:
        t, cost = zip(*s.trace)
        fig.add_scatter(x=t, y=cost, mode="lines", line_shape="hv", name=s.method)
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Runtime [s]",
        yaxis_title="Total distance [km]",
        xaxis_type="log",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


def tradeoff(rows):
    k, dist = zip(*rows)
    fig = go.Figure(go.Scatter(x=k, y=dist, mode="lines+markers"))
    fig.update_layout(
        template="plotly_white", xaxis_title="Vehicles", yaxis_title="Total distance [km]"
    )
    return fig


def to_readme(df, tag):
    rows = [df.columns, ["---"] * df.shape[1], *df.astype(str).to_numpy()]
    table = "\n".join("| " + " | ".join(map(str, r)) + " |" for r in rows)
    block = f"<!-- {tag} -->\n{table}\n<!-- /{tag} -->"
    text = README.read_text()
    README.write_text(
        re.sub(f"<!-- {tag} -->.*?<!-- /{tag} -->", lambda _: block, text, flags=re.S)
    )


def robustness(summary):
    fig = go.Figure()
    for col, name in [("stops_on_time_pct", "Stops on time"), ("days_on_time_pct", "Days on time")]:
        fig.add_scatter(
            x=summary.buffer_pct,
            y=summary[col],
            mode="lines+markers+text",
            name=name,
            text=[f"+{d} % km" for d in summary.extra_km_pct] if col.startswith("days") else None,
            textposition=["bottom right", "top left"] * len(summary),
            cliponaxis=False,
        )
    fig.update_layout(
        template="plotly_white",
        xaxis_title="Planning buffer on travel times [%]",
        yaxis_title="Share on time [%]",
        legend=dict(orientation="h", y=1.1),
        margin=dict(r=90),
    )
    return fig
