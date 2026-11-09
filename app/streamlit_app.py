import streamlit as st
from streamlit_folium import st_folium

from cityroute import heuristics, solve, viz
from cityroute.network import Network
from cityroute.scenario import city

PLACE = "Heidelberg, Germany"
METHODS = ["OR-Tools (Guided Local Search)", "Clarke-Wright Savings", "Nearest Neighbor"]

st.set_page_config("City Route Optimizer", layout="wide")


@st.cache_resource
def network():
    return Network(PLACE)


@st.cache_data
def scenario(customers, vehicles, capacity, seed):
    return city(network(), customers, vehicles, capacity, seed)


@st.cache_data
def run(sc_key, method, windows, limit):
    sc = scenario(*sc_key)
    if method == METHODS[1]:
        return heuristics.savings(sc)
    if method == METHODS[2]:
        return heuristics.nearest_neighbor(sc)
    return solve.ortools(sc, limit, windows)


with st.sidebar:
    st.header("Scenario")
    customers = st.slider("Customers", 10, 50, 30)
    vehicles = st.slider("Vehicles", 1, 15, 5)
    capacity = st.slider("Capacity per vehicle", 10, 100, 50)
    seed = st.number_input("Seed", 0, 999, 0)
    st.header("Solver")
    method = st.selectbox("Method", METHODS)
    windows = st.toggle("Time windows", True, disabled=method != METHODS[0])
    limit = st.slider("Time limit [s]", 1, 30, 5, disabled=method != METHODS[0])

st.title(f"Vehicle routing in {PLACE.split(',')[0]}")
key = (customers, vehicles, capacity, seed)
sc = scenario(*key)
windows = windows and method == METHODS[0]
sol = run(key, method, windows, limit)

if sol is None:
    st.error("No feasible solution. Add vehicles, raise capacity or relax time windows.")
    st.stop()

a, b, c, d = st.columns(4)
a.metric("Total distance", f"{sol.distance:.1f} km")
b.metric("Vehicles used", f"{sol.vehicles} / {vehicles}")
c.metric("Total demand", f"{sc.demand.sum()} / {vehicles * capacity}")
d.metric("Runtime", f"{sol.runtime:.2f} s")
if sol.vehicles > vehicles:
    st.warning("The heuristic ignores the fleet size and needs more vehicles than available.")

st_folium(
    viz.route_map(network(), sc, sol, windows),
    height=620,
    use_container_width=True,
    returned_objects=[],
)
if len(sol.trace) > 1:
    st.plotly_chart(viz.convergence([sol]))
