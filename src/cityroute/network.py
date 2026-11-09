import re
from pathlib import Path

import numpy as np
import osmnx as ox
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from shapely import LineString

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

    def buildings(self):
        path = DATA / f"{_slug(self.place)}_buildings.npy"
        if not path.exists():
            c = ox.features_from_place(self.place, {"building": True}).geometry.centroid
            path.parent.mkdir(exist_ok=True)
            np.save(path, np.column_stack([c.x, c.y]))
        return np.load(path)

    def matrices(self, sites):
        dist, pred = dijkstra(self.length, indices=sites, return_predecessors=True)
        time = dijkstra(self.time, indices=sites)
        return dist[:, sites], time[:, sites], pred

    def path(self, pred, sites, i, j):
        nodes = [sites[j]]
        while nodes[-1] != sites[i]:
            nodes.append(pred[i, nodes[-1]])
        return nodes[::-1]

    def line(self, nodes, tolerance=0.0):
        parts = [self.edges[a, b][:-1] for a, b in zip(nodes[:-1], nodes[1:])]
        coords = np.vstack([*parts, self.xy[nodes[-1:]]])
        return np.asarray(LineString(coords).simplify(tolerance).coords) if tolerance else coords
