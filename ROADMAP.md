# Roadmap

1. [x] Setup: `uv sync`, tests green
2. [x] Network + scenario for Heidelberg cached in `data/`
3. [x] City benchmark (fastest-path matrices): customers on street edges; CVRP HGS = GLS 99.49 km, VRPTW HGS = GLS 129.16 km with 6 vehicles, both proven optimal; trade-off rises monotonically from 6 vehicles (~1.8 km per extra vehicle); visuals checked
4. [x] Solomon: all 56 instances vs SINTEF BKS; HGS within 0.6 % on average, 3 vehicles above BKS in total, GLS 26
5. [x] Streamlit app runs locally (checked in headless Chrome)
6. [ ] Streamlit Community Cloud deployment (requirements/packaging)
7. [x] README with results: tables are written by the scripts in `benchmarks/`
8. [x] PyVRP (HGS) vs OR-Tools: HGS best on 30/30 city instances, GLS +3.2 % at 100 customers
9. [x] Exact reference: CP-SAT proves the optimum up to 60 customers, bound within 3.7 % at 100
10. [~] Extensions: stochastic (`robust.py`) and time-of-day travel times (`traffic.py`) done; open: dynamic orders, CO2 per tour, learned heuristic
