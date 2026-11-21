"""Policies, episode recording and evaluation for Courier AI.

A "policy" is a function that takes the environment and returns a move (0=U, 1=D, 2=L, 3=R).
"""
import json
from env import CourierEnv, astar, astar_action, state, run_episode


def load_q(path="qtable.json"):
    """Load the trained Q-table. Keys are saved as 'a,d0,d1,d2,d3' strings."""
    raw = json.load(open(path))
    return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}


def astar_policy(env):
    """Baseline: always follow the shortest path. Ignores traffic."""
    return astar_action(env)[0]


def make_q_policy(Q):
    """A* + Q-learning: pick the move with the highest learned value.
    If the situation was never seen in training, fall back to A*'s move."""
    def policy(env):
        q = Q.get(state(env))
        if q is None or max(q) == min(q):
            return astar_action(env)[0]
        return max(range(4), key=lambda k: q[k])
    return policy


def play(seed, n_hazards, policy):
    """Run one episode and record every tick so the app can replay it.
    Returns (outcome, steps, walls, goal, frames)."""
    env = CourierEnv(seed, n_hazards)
    frames = []
    outcome = None
    while outcome is None:
        move = policy(env)
        frames.append({"t": env.t, "pos": env.pos, "haz": list(env.haz),
                       "path": astar(env.walls, env.pos, env.goal) or [],
                       "move": move, "astar_move": astar_action(env)[0]})
        outcome = env.step(move)
    frames.append({"t": env.t, "pos": env.pos, "haz": list(env.haz), "path": [],
                   "move": None, "astar_move": None})
    return outcome, env.t, env.walls, env.goal, frames


def evaluate(policy, seeds, n_hazards=3):
    """Run the policy on many maps. Returns delivery/crash/timeout rates and average
    steps for delivered runs."""
    counts = {"delivered": 0, "crash": 0, "timeout": 0}
    steps = []
    seeds = list(seeds)
    for s in seeds:
        env = CourierEnv(s, n_hazards)
        outcome, t = run_episode(env, policy)
        counts[outcome] += 1
        if outcome == "delivered":
            steps.append(t)
    n = len(seeds)
    return {"delivered": counts["delivered"] / n, "crash": counts["crash"] / n,
            "timeout": counts["timeout"] / n,
            "avg_steps": sum(steps) / len(steps) if steps else None}
