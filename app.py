import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import pandas as pd
import streamlit as st

from env import N, DNAME
from agents import load_q, astar_policy, make_q_policy, play, evaluate

st.set_page_config(page_title="Courier AI", layout="wide")

AGENTS = ["A* only", "A* + Q-learning"]
RESULT_TEXT = {"delivered": "DELIVERED", "crash": "CRASHED into traffic", "timeout": "TIMED OUT (60 ticks)"}


@st.cache_resource
def get_q():
    return load_q("qtable.json")


def get_policy(name):
    return astar_policy if name == "A* only" else make_q_policy(get_q())


@st.cache_data(show_spinner=False)
def compare(n_maps, n_hazards):
    """Both agents on the same unseen maps (seeds 10000+, never used in training)."""
    seeds = range(10000, 10000 + n_maps)
    rows = {}
    for name in AGENTS:
        r = evaluate(get_policy(name), seeds, n_hazards)
        rows[name] = {"Delivered %": 100 * r["delivered"], "Crashed %": 100 * r["crash"],
                      "Timed out %": 100 * r["timeout"],
                      "Avg steps (delivered)": r["avg_steps"]}
    return pd.DataFrame(rows).T


def draw(frame, walls, goal):
    fig, ax = plt.subplots(figsize=(4.6, 4.6), dpi=110)
    ax.set_xlim(0, N); ax.set_ylim(N, 0); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for (r, c) in walls:
        ax.add_patch(Rectangle((c, r), 1, 1, fc="#555555"))
    for (r, c) in frame["path"]:
        ax.add_patch(Rectangle((c + .32, r + .32), .36, .36, fc="#9dc3e6"))
    ax.add_patch(Rectangle((0, 0), 1, 1, fc="#70ad47", alpha=.6))
    ax.add_patch(Rectangle((goal[1], goal[0]), 1, 1, fc="#ed7d31"))
    ax.text(goal[1] + .5, goal[0] + .55, "D", ha="center", va="center", color="white", fontsize=9)
    for (r, c) in frame["haz"]:
        ax.add_patch(Circle((c + .5, r + .5), .36, fc="#c0504d"))
    r, c = frame["pos"]
    ax.add_patch(Circle((c + .5, r + .5), .3, fc="#1f4e79", ec="white", lw=1.5))
    for i in range(N + 1):
        ax.plot([0, N], [i, i], color="#dddddd", lw=.4)
        ax.plot([i, i], [0, N], color="#dddddd", lw=.4)
    fig.tight_layout(pad=0.2)
    return fig


def show(placeholder, frames, i, walls, goal):
    fig = draw(frames[i], walls, goal)
    placeholder.pyplot(fig)
    plt.close(fig)


st.title("Courier AI")
st.caption("Watch A* search and Q-learning deliver a package across a city with moving traffic.")
st.warning("Simulation only: the map is generated and the traffic moves randomly. "
           "This is not a real routing or vehicle-control tool.")

with st.sidebar:
    st.header("Input")
    seed = st.number_input("Map seed", min_value=0, max_value=999999, value=10003, step=1)
    n_haz = st.slider("Traffic hazards", 1, 6, 3)
    agent = st.radio("Agent", AGENTS)
    run = st.button("Run episode", type="primary", width="stretch")
    st.divider()
    n_maps = st.number_input("Maps to compare agents on", min_value=10, max_value=500, value=100, step=10)
    do_compare = st.button("Compare both agents", width="stretch")

if run:
    try:
        outcome, steps, walls, goal, frames = play(int(seed), n_haz, get_policy(agent))
        st.session_state["run"] = dict(outcome=outcome, steps=steps, walls=walls, goal=goal,
                                       frames=frames, agent=agent, seed=int(seed), n_haz=n_haz)
    except FileNotFoundError:
        st.error("qtable.json not found. Run `python train.py` once to create it.")

left, right = st.columns([1.1, 1])
if "run" in st.session_state:
    R = st.session_state["run"]
    frames = R["frames"]
    with left:
        grid = st.empty()
        i = st.slider("Tick", 0, len(frames) - 1, 0)
        play_btn = st.button("Play")
        if play_btn:
            for k in range(len(frames)):
                show(grid, frames, k, R["walls"], R["goal"])
                time.sleep(0.3)
        else:
            show(grid, frames, i, R["walls"], R["goal"])
        st.caption("Blue dots: A* path from the courier's current cell. Red: traffic. Dark circle: courier.")
    with right:
        st.subheader("Output")
        st.markdown(f"**Result: {RESULT_TEXT[R['outcome']]}**")
        st.write(f"Agent: {R['agent']}  |  Map seed: {R['seed']}  |  Hazards: {R['n_haz']}")
        st.write(f"Steps taken: {R['steps']}")
        f = frames[i]
        if f["move"] is not None:
            same = "same as A*" if f["move"] == f["astar_move"] else "different from A*"
            st.write(f"Tick {f['t']}: move **{DNAME[f['move']]}** (A* says {DNAME[f['astar_move']]}, {same})")
        else:
            st.write(f"Tick {f['t']}: end of episode")
else:
    left.info("Choose a map seed, hazards and an agent, then press **Run episode**.")

if do_compare:
    try:
        with st.spinner(f"Running both agents on {int(n_maps)} unseen maps..."):
            st.session_state["cmp"] = (int(n_maps), n_haz, compare(int(n_maps), n_haz))
    except FileNotFoundError:
        st.error("qtable.json not found. Run `python train.py` once to create it.")

if "cmp" in st.session_state:
    n, h, df = st.session_state["cmp"]
    st.divider()
    st.subheader(f"Comparison over {n} unseen maps ({h} hazards)")
    st.dataframe(df.round(1), width="stretch")
    st.bar_chart(df[["Delivered %", "Crashed %", "Timed out %"]])
    if h != 3:
        st.caption("The Q-learning agent was trained with 3 hazards, so other counts test how well it generalizes.")
