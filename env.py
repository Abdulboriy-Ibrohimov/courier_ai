import random, heapq
from collections import defaultdict

N = 10
MAXT = 60
DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # U D L R
DNAME = ["U", "D", "L", "R"]


class CourierEnv:
    """10x10 city grid. Courier goes S -> D. 3 traffic hazards move randomly each tick."""

    def __init__(self, seed):
        self.rng = random.Random(seed)
        while True:
            self.walls = {(r, c) for r in range(N) for c in range(N) if self.rng.random() < 0.15}
            self.start, self.goal = (0, 0), (N - 1, N - 1)
            self.walls -= {self.start, self.goal}
            if astar(self.walls, self.start, self.goal):
                break
        free = [(r, c) for r in range(N) for c in range(N)
                if (r, c) not in self.walls and abs(r - c) > 3 and (r, c) not in (self.start, self.goal)]
        self.haz = self.rng.sample(free, 3)
        self.pos = self.start
        self.t = 0

    def free(self, p):
        return 0 <= p[0] < N and 0 <= p[1] < N and p not in self.walls

    def step(self, a):
        nxt = (self.pos[0] + DIRS[a][0], self.pos[1] + DIRS[a][1])
        if self.free(nxt):
            self.pos = nxt
        self.t += 1
        new = []
        for h in self.haz:
            opts = [h] + [(h[0] + d[0], h[1] + d[1]) for d in DIRS if self.free((h[0] + d[0], h[1] + d[1]))]
            new.append(self.rng.choice(opts))
        self.haz = new
        if self.pos in self.haz:
            return "crash"
        if self.pos == self.goal:
            return "delivered"
        if self.t >= MAXT:
            return "timeout"
        return None


def astar(walls, s, g):
    openh = [(abs(s[0] - g[0]) + abs(s[1] - g[1]), 0, s)]
    best = {s: 0}
    parent = {}
    while openh:
        _, gc, p = heapq.heappop(openh)
        if p == g:
            path = [p]
            while p in parent:
                p = parent[p]
                path.append(p)
            return path[::-1]
        for d in DIRS:
            q = (p[0] + d[0], p[1] + d[1])
            if 0 <= q[0] < N and 0 <= q[1] < N and q not in walls and gc + 1 < best.get(q, 1e9):
                best[q] = gc + 1
                parent[q] = p
                heapq.heappush(openh, (gc + 1 + abs(q[0] - g[0]) + abs(q[1] - g[1]), gc + 1, q))
    return None


def astar_action(env):
    path = astar(env.walls, env.pos, env.goal)
    if not path or len(path) < 2:
        return 0, 0
    d = (path[1][0] - path[0][0], path[1][1] - path[0][1])
    return DIRS.index(d), len(path) - 1


def danger(env, a):
    nxt = (env.pos[0] + DIRS[a][0], env.pos[1] + DIRS[a][1])
    if not env.free(nxt):
        return 1  # blocked
    near = any(abs(nxt[0] - h[0]) + abs(nxt[1] - h[1]) <= 1 for h in env.haz)
    return 2 if near else 0


def state(env):
    a, _ = astar_action(env)
    return (a,) + tuple(danger(env, k) for k in range(4))


def run_episode(env, policy):
    out = None
    while out is None:
        out = env.step(policy(env))
    return out, env.t
