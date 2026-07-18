import sys
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


MOVES = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
GHOST_MOVES = MOVES + (Move.STAY,)
INF = 10 ** 6


class _GraphAgent:
    def _init_graph(self):
        self._ready = False
        self._shape = None
        self._tiles = ()
        self._index = {}
        self._neighbors = ()
        self._dist = ()
        self._degree = ()
        self._dead_depth = ()
        self._last_enemy = None
        self._history = deque(maxlen=10)

    def _prepare(self, map_state):
        shape = tuple(map_state.shape)
        if self._ready and shape == self._shape:
            return
        self._shape = shape
        self._tiles = tuple(map(tuple, np.argwhere(map_state != 1)))
        self._index = {p: i for i, p in enumerate(self._tiles)}
        n = len(self._tiles)
        nbr = []
        for r, c in self._tiles:
            row = []
            for move in MOVES:
                q = (r + move.value[0], c + move.value[1])
                if q in self._index:
                    row.append(self._index[q])
            nbr.append(tuple(row))
        self._neighbors = tuple(nbr)
        self._degree = tuple(len(x) for x in nbr)

        # All-pairs unweighted graph distances; the map has fewer than 250 tiles.
        all_dist = []
        for source in range(n):
            d = [INF] * n
            d[source] = 0
            queue = deque([source])
            while queue:
                u = queue.popleft()
                nd = d[u] + 1
                for v in nbr[u]:
                    if d[v] == INF:
                        d[v] = nd
                        queue.append(v)
            all_dist.append(tuple(d))
        self._dist = tuple(all_dist)

        # Depth in a pruned corridor/dead-end branch (0 means safe core).
        degree = [len(x) for x in nbr]
        removed = [False] * n
        depth = [0] * n
        queue = deque(i for i, value in enumerate(degree) if value <= 1)
        while queue:
            u = queue.popleft()
            if removed[u]:
                continue
            removed[u] = True
            for v in nbr[u]:
                if not removed[v]:
                    depth[u] = max(depth[u], depth[v] + 1)
                    degree[v] -= 1
                    if degree[v] <= 1:
                        queue.append(v)
        self._dead_depth = tuple(depth)
        self._ready = True

    def _ghost_options(self, pos):
        result = []
        r, c = pos
        for move in GHOST_MOVES:
            q = (r + move.value[0], c + move.value[1])
            if q in self._index:
                result.append((move, q))
        return result

    def _pacman_options(self, pos):
        result = []
        r, c = pos
        for move in MOVES:
            dr, dc = move.value
            q = (r, c)
            for steps in range(1, self.pacman_speed + 1):
                nxt = (q[0] + dr, q[1] + dc)
                if nxt not in self._index:
                    break
                q = nxt
                result.append((move, steps, q))
        if not result:
            result.append((Move.STAY, 1, pos))
        return result

    @staticmethod
    def _captured(pacman, ghost):
        return abs(pacman[0] - ghost[0]) + abs(pacman[1] - ghost[1]) < 2

    def _d(self, a, b):
        return self._dist[self._index[a]][self._index[b]]


class PacmanAgent(BasePacmanAgent, _GraphAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._init_graph()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._prepare(map_state)
        if enemy_position is not None:
            self._last_enemy = tuple(enemy_position)
        ghost = self._last_enemy
        pacman = tuple(my_position)
        self._history.append(pacman)
        if ghost is None or ghost not in self._index:
            return self._explore(pacman)

        @lru_cache(maxsize=None)
        def value(p, g, depth):
            if self._captured(p, g):
                return -10000 - depth * 100
            if depth == 0:
                distance = self._d(p, g)
                gi = self._index[g]
                # Low is good: distance plus the Ghost's local escape quality.
                return distance * 30 + self._degree[gi] * 7 - self._dead_depth[gi] * 9
            best = INF
            for _, _, np_ in self._pacman_options(p):
                worst = -INF
                for _, ng in self._ghost_options(g):
                    score = value(np_, ng, depth - 1)
                    if score > worst:
                        worst = score
                    if worst >= best:
                        break
                if worst < best:
                    best = worst
            return best

        best_action = None
        best_key = None
        repeated = set(self._history)
        for move, steps, landing in self._pacman_options(pacman):
            worst = -INF
            captures = 0
            for _, ng in self._ghost_options(ghost):
                if self._captured(landing, ng):
                    captures += 1
                score = value(landing, ng, 2)
                worst = max(worst, score)
            # Prefer robust captures, then minimax value, longer useful sprints,
            # and non-repeated positions.
            key = (-captures, worst, landing in repeated, -steps)
            if best_key is None or key < best_key:
                best_key = key
                best_action = (move, steps)
        return best_action or (Move.STAY, 1)

    def _explore(self, pos):
        seen = set(self._history)
        options = self._pacman_options(pos)
        _, _, target = max(options, key=lambda x: (x[2] not in seen, self._degree[self._index[x[2]]], x[1]))
        for move, steps, landing in options:
            if landing == target:
                return move, steps
        return Move.STAY, 1


class GhostAgent(BaseGhostAgent, _GraphAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # The loader does not forward pacman_speed to Ghost; tournament default is 2.
        self.pacman_speed = 2
        self._init_graph()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._prepare(map_state)
        if enemy_position is not None:
            self._last_enemy = tuple(enemy_position)
        pacman = self._last_enemy
        ghost = tuple(my_position)
        self._history.append(ghost)
        if pacman is None or pacman not in self._index:
            return self._safe_wander(ghost)

        # Stable distance/mobility cycles are strategically valuable for an
        # evader.  Use this policy whenever every immediate Pacman reply is
        # non-capturing; the deeper safety search below handles close combat.
        cycle_choices = []
        for order, (move, landing) in enumerate(self._ghost_options(ghost)):
            exposed = any(
                self._captured(np_, landing)
                for _, _, np_ in self._pacman_options(pacman)
            )
            if not exposed:
                li = self._index[landing]
                cycle_choices.append((
                    self._d(pacman, landing), self._degree[li],
                    -self._dead_depth[li], -order, move,
                ))
        if cycle_choices and self._d(pacman, ghost) >= 5:
            return max(cycle_choices)[-1]

        @lru_cache(maxsize=None)
        def value(p, g, depth):
            if self._captured(p, g):
                return -10000 - depth * 100
            gi = self._index[g]
            if depth == 0:
                distance = self._d(p, g)
                return distance * 35 + self._degree[gi] * 12 - self._dead_depth[gi] * 24
            best = -INF
            for _, ng in self._ghost_options(g):
                worst = INF
                for _, _, np_ in self._pacman_options(p):
                    score = value(np_, ng, depth - 1)
                    if score < worst:
                        worst = score
                    if worst <= best:
                        break
                best = max(best, worst)
            return best

        repeated = set(self._history)
        best_move = Move.STAY
        best_key = None
        for move, landing in self._ghost_options(ghost):
            worst = INF
            immediate_losses = 0
            for _, _, np_ in self._pacman_options(pacman):
                if self._captured(np_, landing):
                    immediate_losses += 1
                worst = min(worst, value(np_, landing, 2))
            li = self._index[landing]
            key = (immediate_losses, -worst, self._dead_depth[li], landing in repeated, -self._degree[li], move == Move.STAY)
            if best_key is None or key < best_key:
                best_key = key
                best_move = move
        return best_move

    def _safe_wander(self, pos):
        seen = set(self._history)
        return min(
            self._ghost_options(pos),
            key=lambda x: (self._dead_depth[self._index[x[1]]], x[1] in seen,
                           -self._degree[self._index[x[1]]], x[0] == Move.STAY),
        )[0]
