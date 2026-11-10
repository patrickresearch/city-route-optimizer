import re
from collections import defaultdict
from copy import copy
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import LineString, STRtree, line_locate_point, points
from shapely.ops import substring

DATA = Path(__file__).resolve().parents[2] / "data"


def _slug(place):
    return re.sub(r"\W+", "_", place.lower()).strip("_")


def _graph(place):
    path = DATA / f"{_slug(place)}.graphml"
    if path.exists():
        return ox.load_graphml(path)
    g = ox.graph_from_place(place, network_type="drive")
    g = ox.truncate.largest_component(g, strongly=True)
    g = ox.routing.add_edge_travel_times(ox.routing.add_edge_speeds(g))
    path.parent.mkdir(exist_ok=True)
    ox.save_graphml(g, path)
    return g


def shortest_paths(time, length, sites):
    t, pred = dijkstra(time, indices=sites, return_predecessors=True)
    rows, cols = np.indices(pred.shape)
    root = pred < 0
    parent = np.where(root, cols, pred)
    dist = np.asarray(length[parent.ravel(), cols.ravel()]).reshape(pred.shape)
    dist[root] = 0
    while (parent != parent[rows, parent]).any():
        dist = dist + dist[rows, parent]
        parent = parent[rows, parent]
    return dist[:, sites], t[:, sites], pred


def attach(xy, length, time, edges, aspect, stops):
    keys = [k for k in edges if k[0] != k[1]]
    lines = np.array([LineString(edges[k] * aspect) for k in keys])
    pts = points(np.asarray(stops) * aspect)
    hit = STRtree(lines).nearest(pts)
    frac = np.clip(line_locate_point(lines[hit], pts, normalized=True), 0.001, 0.999)
    n = len(xy)
    groups = defaultdict(list)
    for s, (h, f) in enumerate(zip(hit, frac)):
        u, v = keys[h]
        both = (v, u) in edges and np.isclose(length[v, u], length[u, v])
        if both and u > v:
            u, v, f = v, u, 1 - f
        groups[u, v, both].append((f, n + s))

    edges, new, arcs = dict(edges), np.empty((len(stops), 2)), []
    for (u, v, both), members in groups.items():
        members.sort()
        line = LineString(edges[u, v] * aspect)
        cuts = [0.0, *(f for f, _ in members), 1.0]
        chain = [u, *(s for _, s in members), v]
        for f, s in members:
            new[s - n] = np.asarray(line.interpolate(f, normalized=True).coords[0]) / aspect
        for a, b, lo, hi in zip(chain[:-1], chain[1:], cuts[:-1], cuts[1:]):
            part = np.asarray(substring(line, lo, hi, normalized=True).coords) / aspect
            for src, dst, geom, key in [(a, b, part, (u, v)), (b, a, part[::-1], (v, u))][
                : 1 + both
            ]:
                edges[src, dst] = geom
                arcs.append((src, dst, (hi - lo) * length[key], (hi - lo) * time[key]))

    rows, cols, lens, times = np.array(arcs).T
    size = n + len(stops)

    def grow(m, extra):
        m = m.tocoo()
        data = np.r_[m.data, np.maximum(extra, 1e-9)]
        return csr_matrix((data, (np.r_[m.row, rows], np.r_[m.col, cols])), shape=(size, size))

    return np.vstack([xy, new]), grow(length, lens), grow(time, times), edges, np.arange(n, size)


class Network:
    def __init__(self, place="Heidelberg, Germany"):
        self.place = place
        self.graph = _graph(place)
        nodes, edges = ox.graph_to_gdfs(self.graph)
        n = len(nodes)
        self.xy = nodes[["x", "y"]].to_numpy()
        pos = pd.Series(np.arange(n), index=nodes.index)
        best = edges.loc[edges.groupby(level=[0, 1])["length"].idxmin()]
        u = pos[best.index.get_level_values(0)].to_numpy()
        v = pos[best.index.get_level_values(1)].to_numpy()
        self.length = csr_matrix((best["length"].to_numpy(), (u, v)), shape=(n, n))
        self.time = csr_matrix((best["travel_time"].to_numpy(), (u, v)), shape=(n, n))
        self.edges = {(a, b): np.asarray(g.coords) for a, b, g in zip(u, v, best.geometry)}
        self._aspect = np.array([np.cos(np.radians(self.xy[:, 1].mean())), 1.0])
        self._tree = cKDTree(self.xy * self._aspect)

    def nearest(self, lonlat):
        return self._tree.query(np.asarray(lonlat) * self._aspect)[1]

    def attach(self, stops):
        net = copy(self)
        net.xy, net.length, net.time, net.edges, sites = attach(
            self.xy, self.length, self.time, self.edges, self._aspect, stops
        )
        return net, sites

    def buildings(self):
        path = DATA / f"{_slug(self.place)}_buildings.npy"
        if not path.exists():
            b = ox.projection.project_gdf(ox.features_from_place(self.place, {"building": True}))
            c = b.geometry.centroid.to_crs(4326)
            path.parent.mkdir(exist_ok=True)
            np.save(path, np.column_stack([c.x, c.y]))
        return np.load(path)

    def matrices(self, sites):
        return shortest_paths(self.time, self.length, sites)

    def path(self, pred, sites, i, j):
        nodes = [sites[j]]
        while nodes[-1] != sites[i]:
            nodes.append(pred[i, nodes[-1]])
        return nodes[::-1]

    def line(self, nodes, tolerance=0.0):
        parts = [self.edges[a, b][:-1] for a, b in zip(nodes[:-1], nodes[1:])]
        coords = np.vstack([*parts, self.xy[nodes[-1:]]])
        return np.asarray(LineString(coords).simplify(tolerance).coords) if tolerance else coords
