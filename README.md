# City Route Optimizer

Vehicle routing with capacities and time windows on the real street network of Heidelberg.
A delivery fleet leaves a central depot, serves customers placed on OpenStreetMap buildings and
returns, minimising total driven distance.

![routes](assets/routes.gif)

## Approach

1. Drivable street network from OpenStreetMap (`osmnx`), restricted to its largest strongly connected component.
2. Customers sampled from building centroids and snapped to the nearest intersection, each with a demand and a one- or two-hour delivery window.
3. Distance and travel-time matrices via Dijkstra on a sparse adjacency matrix (`scipy.sparse.csgraph`), one source per site.
4. Solvers: Nearest Neighbor, Clarke-Wright Savings and OR-Tools with greedy descent, Guided Local Search, Simulated Annealing and Tabu Search.
5. Routes are mapped back onto the streets, simplified with Douglas-Peucker and animated.

## Results

| Problem | Method | Distance [km] | Vehicles | Runtime [s] |
|---|---|---|---|---|
| | | | | |

![convergence](assets/convergence.png)
![tradeoff](assets/tradeoff.png)

## Usage

```bash
uv sync
uv run pytest
uv run python benchmarks/city.py
uv run streamlit run app/streamlit_app.py
```

Solomon benchmark: place instance files (e.g. `c101.txt`) in `benchmarks/solomon/`, then

```bash
uv run python benchmarks/solomon.py
```

## Structure

```
src/cityroute/   network, scenario, solve, heuristics, viz
app/             Streamlit what-if planner
benchmarks/      city comparison, Solomon instances
tests/           feasibility and solver checks
assets/          figures and animation
```

## License

MIT
