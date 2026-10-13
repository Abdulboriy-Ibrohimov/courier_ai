import random, json
from collections import defaultdict
from env import *

TEST_SEEDS = range(10000, 10500)
TRAIN_SEEDS = range(0, 3000)


def evaluate(policy_factory, seeds):
    res = defaultdict(int)
    steps = []
    for s in seeds:
        env = CourierEnv(s)
        pol = policy_factory(env)
        o, t = run_episode(env, pol)
        res[o] += 1
        if o == "delivered":
            steps.append(t)
    n = len(list(seeds))
    return {k: res[k] / n for k in ("delivered", "crash", "timeout")}, (sum(steps) / len(steps) if steps else None)


rand_pol = lambda env: (lambda e: random.randrange(4))
astar_pol = lambda env: (lambda e: astar_action(e)[0])

print("random", evaluate(rand_pol, TEST_SEEDS))
print("astar_only", evaluate(astar_pol, TEST_SEEDS))

# Q-learning sanity check
Q = defaultdict(lambda: [0.0] * 4)
alpha, gamma = 0.2, 0.95
curve = []
rng = random.Random(1)
EP = 20000
window = []
first_rewards = defaultdict(int)
for ep in range(EP):
    eps = max(0.05, 1 - ep / (EP * 0.6))
    env = CourierEnv(rng.choice(list(TRAIN_SEEDS)))
    s = state(env)
    out = None
    while out is None:
        a = rng.randrange(4) if rng.random() < eps else max(range(4), key=lambda k: Q[s][k])
        before = astar_action(env)[1]
        out = env.step(a)
        after = astar_action(env)[1] if out is None else 0
        if out == "delivered":
            r = 10
        elif out == "crash":
            r = -10
        else:
            r = -0.1 + 0.3 * (before - after)
        s2 = state(env) if out is None else None
        target = r + (gamma * max(Q[s2]) if s2 is not None else 0)
        Q[s][a] += alpha * (target - Q[s][a])
        s = s2
    window.append(1 if out == "delivered" else 0)
    if (ep + 1) % 500 == 0:
        curve.append((ep + 1, sum(window[-500:]) / 500))

def q_pol(env):
    return lambda e: max(range(4), key=lambda k: Q[state(e)][k])

print("q_states", len(Q))
print("q_test", evaluate(q_pol, TEST_SEEDS))
json.dump(curve, open("curve.json", "w"))
print(curve)
