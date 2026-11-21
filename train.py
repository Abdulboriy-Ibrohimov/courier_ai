"""Train the Q-learning agent and save the Q-table to qtable.json.

Same training as the prototype: 20,000 episodes on maps from seeds 0-2999, 3 hazards.
Takes about 2 minutes. Run:  python train.py
"""
import json, random
from collections import defaultdict
from env import CourierEnv, astar_action, state

TRAIN_SEEDS = list(range(0, 3000))
EPISODES = 20000
ALPHA, GAMMA = 0.2, 0.95          # learning rate, discount
EPS_START, EPS_END = 1.0, 0.05    # exploration: random moves early, mostly greedy later

Q = defaultdict(lambda: [0.0] * 4)
rng = random.Random(1)

for ep in range(EPISODES):
    eps = max(EPS_END, EPS_START - ep / (EPISODES * 0.6))
    env = CourierEnv(rng.choice(TRAIN_SEEDS))
    s = state(env)
    outcome = None
    while outcome is None:
        a = rng.randrange(4) if rng.random() < eps else max(range(4), key=lambda k: Q[s][k])
        before = astar_action(env)[1]                 # A* distance to the depot before the move
        outcome = env.step(a)
        after = astar_action(env)[1] if outcome is None else 0
        if outcome == "delivered":
            r = 10
        elif outcome == "crash":
            r = -10
        else:
            r = -0.1 + 0.3 * (before - after)         # step cost + progress reward from A*
        s2 = state(env) if outcome is None else None
        target = r + (GAMMA * max(Q[s2]) if s2 is not None else 0)
        Q[s][a] += ALPHA * (target - Q[s][a])         # the Q-learning update
        s = s2
    if (ep + 1) % 2000 == 0:
        print("episode", ep + 1, "states seen", len(Q), flush=True)

json.dump({",".join(map(str, k)): v for k, v in Q.items()}, open("qtable.json", "w"))
print("saved qtable.json with", len(Q), "states")
