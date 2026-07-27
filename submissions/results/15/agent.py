"""
Hide and Seek Arena - Limited Vision Agent

PacmanAgent (Seeker):
  Speed-aware A* toward ghost (or last known pos).
  1-ply intercept lookahead when enemy is visible.
  Frontier BFS exploration when enemy has never been seen.

GhostAgent (Hider):
  Voronoi evasion (speed-aware pacman BFS vs ghost BFS) when enemy visible.
  Navigates to safest hiding cell when enemy not visible.
  Memory: tracks last known enemy position across steps.
"""

import sys
import heapq
from collections import deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent, GhostAgent as BaseGhostAgent
from environment import Move

_DIRS       = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_GHOST_DIRS = [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]
_INF        = 441


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

def _valid(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    # -1 = unseen but potentially passable; 1 = confirmed wall
    return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1


def _bfs_dist(map_state, start, dirs=None):
    """BFS from start. Returns {pos: steps} for every reachable cell."""
    if dirs is None:
        dirs = _DIRS
    dist = {start: 0}
    q = deque([start])
    while q:
        p = q.popleft()
        for mv in dirs:
            dr, dc = mv.value
            nxt = (p[0] + dr, p[1] + dc)
            if nxt not in dist and _valid(nxt, map_state):
                dist[nxt] = dist[p] + 1
                q.append(nxt)
    return dist


def _bfs_speed_dist(map_state, start, max_speed):
    """Speed-aware BFS: each step covers 1..max_speed cells straight.
    Returns {pos: min_game_steps} — exact seeker cost to every cell."""
    dist = {start: 0}
    q = deque([start])
    while q:
        p = q.popleft()
        for mv in _DIRS:
            dr, dc = mv.value
            for n in range(1, max_speed + 1):
                nxt = (p[0] + dr * n, p[1] + dc * n)
                if not _valid(nxt, map_state):
                    break
                if nxt not in dist:
                    dist[nxt] = dist[p] + 1
                    q.append(nxt)
    return dist


def _astar(map_state, start, goal):
    """Standard A* (1-step moves). Returns position list or None."""
    if start == goal:
        return [start]
    h = lambda p: abs(p[0] - goal[0]) + abs(p[1] - goal[1])
    heap = [(h(start), 0, start)]
    came_from = {}
    g = {start: 0}
    while heap:
        _, cost, p = heapq.heappop(heap)
        if p == goal:
            path = [p]
            while p in came_from:
                p = came_from[p]
                path.append(p)
            return list(reversed(path))
        if cost > g.get(p, _INF):
            continue
        for mv in _DIRS:
            dr, dc = mv.value
            nxt = (p[0] + dr, p[1] + dc)
            if not _valid(nxt, map_state):
                continue
            nc = cost + 1
            if nc < g.get(nxt, _INF):
                g[nxt] = nc
                came_from[nxt] = p
                heapq.heappush(heap, (nc + h(nxt), nc, nxt))
    return None


def _astar_first_move(map_state, start, goal):
    """Return just the first Move from A*, or None if unreachable."""
    path = _astar(map_state, start, goal)
    if path and len(path) >= 2:
        dr = path[1][0] - path[0][0]
        dc = path[1][1] - path[0][1]
        for mv in Move:
            if mv.value == (dr, dc):
                return mv
    return None


def _astar_speed(map_state, start, goal, max_speed):
    """Speed-aware A*. Returns (move, steps) for first action."""
    if start == goal:
        return Move.STAY, 1
    h = lambda p: (abs(p[0] - goal[0]) + abs(p[1] - goal[1])) / max_speed
    heap = [(h(start), 0, start)]
    came_from = {}
    g = {start: 0}
    while heap:
        _, cost, p = heapq.heappop(heap)
        if p == goal:
            cur = p
            while came_from[cur][0] != start:
                cur = came_from[cur][0]
            _, first_move, first_steps = came_from[cur]
            return first_move, first_steps
        if cost > g.get(p, _INF):
            continue
        for mv in _DIRS:
            dr, dc = mv.value
            for n in range(1, max_speed + 1):
                nxt = (p[0] + dr * n, p[1] + dc * n)
                if not _valid(nxt, map_state):
                    break
                nc = cost + 1
                if nc < g.get(nxt, _INF):
                    g[nxt] = nc
                    came_from[nxt] = (p, mv, n)
                    heapq.heappush(heap, (nc + h(nxt), nc, nxt))
    return Move.STAY, 1


def _exits(pos, map_state):
    return sum(1 for mv in _DIRS if _valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state))


def _safe_zone_components(safe_zone_set):
    """Connected component sizes within the safe zone."""
    comp_size = {}
    visited = set()
    for start in safe_zone_set:
        if start in visited:
            continue
        component = []
        q = deque([start])
        visited.add(start)
        while q:
            p = q.popleft()
            component.append(p)
            for mv in _DIRS:
                nxt = (p[0] + mv.value[0], p[1] + mv.value[1])
                if nxt in safe_zone_set and nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
        for p in component:
            comp_size[p] = len(component)
    return comp_size


def _intercept_move(map_state, my_pos, enemy_pos, max_speed):
    """1-ply lookahead: pick the move that minimises ghost's safe zone."""
    ghost_dist = _bfs_dist(map_state, enemy_pos)
    best_move, best_steps = Move.STAY, 1
    best_safe, best_tie   = float('inf'), float('inf')

    for mv in _DIRS:
        dr, dc = mv.value
        for n in range(1, max_speed + 1):
            nxt = (my_pos[0] + dr * n, my_pos[1] + dc * n)
            if not _valid(nxt, map_state):
                break
            new_pac   = _bfs_speed_dist(map_state, nxt, max_speed)
            safe_size = sum(1 for pos, gd in ghost_dist.items()
                            if gd <= new_pac.get(pos, _INF))
            tie = abs(nxt[0] - enemy_pos[0]) + abs(nxt[1] - enemy_pos[1])
            if safe_size < best_safe or (safe_size == best_safe and tie < best_tie):
                best_safe, best_tie = safe_size, tie
                best_move, best_steps = mv, n

    return best_move, best_steps


def _random_valid(pos, map_state, dirs=None):
    import random
    if dirs is None:
        dirs = _DIRS
    opts = [mv for mv in dirs if _valid((pos[0] + mv.value[0], pos[1] + mv.value[1]), map_state)]
    return (random.choice(opts) if opts else Move.STAY)


# ---------------------------------------------------------------------------
# PacmanAgent — Seeker
# ---------------------------------------------------------------------------

class PacmanAgent(BasePacmanAgent):
    """
    Visible enemy  → 1-ply intercept with speed-aware A* + leapfrog.
    Last known pos → speed-aware A* to last known position.
    No info        → frontier BFS exploration.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed  = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_enemy    = None
        self.was_visible   = False
        self.visited       = set()
        self.known_map     = None
        self.search_target = None   # pre-computed "likely hiding cell"

    def step(self, map_state, my_position, enemy_position, step_number):
        visible = enemy_position is not None

        # Merge observations
        if self.known_map is None:
            self.known_map = np.full(map_state.shape, -1, dtype=np.int8)
        mask = map_state != -1
        self.known_map[mask] = map_state[mask]

        if visible:
            self.last_enemy    = enemy_position
            self.search_target = None
        elif self.was_visible:
            # Just lost sight — reset visited so we sweep the map fresh
            self.visited.clear()
            self.search_target = self._likely_hiding_cell(self.known_map)
        self.was_visible = visible
        self.visited.add(my_position)

        nav = self.known_map  # use accumulated map for navigation

        nav = self.known_map

        if visible:
            target = self.last_enemy
            move, steps = _intercept_move(nav, my_position, target, self.pacman_speed)

            # Leapfrog: jump one cell past ghost when directly aligned
            dr, dc = move.value
            er = target[0] - my_position[0]
            ec = target[1] - my_position[1]
            same_axis = (dc == 0 and ec == 0 and er * dr > 0) or \
                        (dr == 0 and er == 0 and ec * dc > 0)
            if same_axis:
                gap = abs(er) + abs(ec)
                leap = gap + 1
                if leap <= self.pacman_speed:
                    nxt = (my_position[0] + dr * leap, my_position[1] + dc * leap)
                    if _valid(nxt, nav):
                        steps = leap
            return (move, steps)

        # Not visible — go toward search target or last known, then explore
        goal = self.search_target or self.last_enemy
        if goal is None:
            return self._frontier(my_position, nav)

        if my_position == goal:
            # Arrived — switch to frontier sweep
            return self._frontier(my_position, nav)

        move, steps = _astar_speed(nav, my_position, goal, self.pacman_speed)
        return (move, steps)

    def _likely_hiding_cell(self, nav):
        """Return the cell hardest to reach from any map edge — where a smart ghost hides."""
        h, w = nav.shape
        passable = list(map(tuple, np.argwhere(nav != 1)))
        if not passable:
            return None
        edge_cells = [(r, c) for r, c in passable if r == 0 or r == h-1 or c == 0 or c == w-1]
        if not edge_cells:
            return None
        min_dist = {p: _INF for p in passable}
        for ep in edge_cells:
            d = _bfs_dist(nav, ep)
            for p in passable:
                if d.get(p, _INF) < min_dist[p]:
                    min_dist[p] = d.get(p, _INF)
        reachable = [p for p in passable if min_dist[p] < _INF]
        return max(reachable, key=lambda p: min_dist[p]) if reachable else None

    def _frontier(self, pos, map_state):
        """BFS to nearest unvisited reachable cell."""
        q = deque([(pos, None)])
        seen = {pos}
        while q:
            cur, first = q.popleft()
            if cur not in self.visited and cur != pos:
                move = first
                steps = self._max_steps(pos, move, map_state)
                return (move, steps)
            for mv in _DIRS:
                dr, dc = mv.value
                nxt = (cur[0]+dr, cur[1]+dc)
                if _valid(nxt, map_state) and nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, mv if first is None else first))
        move = _random_valid(pos, map_state)
        return (move, self._max_steps(pos, move, map_state))

    def _max_steps(self, pos, move, map_state):
        steps, cur = 0, pos
        for _ in range(self.pacman_speed):
            dr, dc = move.value
            nxt = (cur[0]+dr, cur[1]+dc)
            if not _valid(nxt, map_state):
                break
            steps += 1
            cur = nxt
        return max(steps, 1)


# ---------------------------------------------------------------------------
# GhostAgent — Hider
# ---------------------------------------------------------------------------

class GhostAgent(BaseGhostAgent):
    """
    Visible enemy  → Voronoi evasion: navigate to best cell in safe zone
                     (maximise pacman_steps * 100 + component_size * 5 + exits * 10).
    Not visible    → head to pre-computed safe hiding cell and stay there.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed_factor  = max(1, int(kwargs.get("pacman_speed", 2)))
        self.last_enemy    = None
        self.hide_target   = None
        self.known_map     = None

    def step(self, map_state, my_position, enemy_position, step_number):
        visible = enemy_position is not None
        if visible:
            self.last_enemy  = enemy_position
            self.hide_target = None
        # Merge observations
        if self.known_map is None:
            self.known_map = np.full(map_state.shape, -1, dtype=np.int8)
        mask = map_state != -1
        self.known_map[mask] = map_state[mask]

        nav = self.known_map
        if visible:
            return self._voronoi_evade(nav, my_position, enemy_position)
        else:
            return self._hide(nav, my_position)

    # ------------------------------------------------------------------
    def _voronoi_evade(self, map_state, my_pos, enemy_pos):
        pacman_steps = _bfs_speed_dist(map_state, enemy_pos, self.speed_factor)
        ghost_dist   = _bfs_dist(map_state, my_pos, dirs=_GHOST_DIRS)

        safe_zone  = {
            pos for pos, gd in ghost_dist.items()
            if gd > 0 and gd <= pacman_steps.get(pos, _INF)
        }
        comp_sizes = _safe_zone_components(safe_zone)

        best_target, best_score = None, -1
        for pos in safe_zone:
            score = (pacman_steps.get(pos, _INF) * 100
                     + comp_sizes.get(pos, 1) * 5
                     + _exits(pos, map_state) * 10)
            if score > best_score:
                best_score, best_target = score, pos

        if best_target is not None:
            move = _astar_first_move(map_state, my_pos, best_target)
            if move is not None:
                return move

        # Fallback: greedy maximise pacman distance
        return self._greedy_evade(map_state, my_pos, pacman_steps)

    def _greedy_evade(self, map_state, my_pos, pacman_dist):
        best_mv    = Move.STAY
        best_score = pacman_dist.get(my_pos, 0) * 100 + _exits(my_pos, map_state) * 10
        for mv in _GHOST_DIRS:
            dr, dc = mv.value
            nxt = (my_pos[0]+dr, my_pos[1]+dc)
            if not _valid(nxt, map_state):
                continue
            score = pacman_dist.get(nxt, 0) * 100 + _exits(nxt, map_state) * 10
            if score > best_score:
                best_score, best_mv = score, mv
        return best_mv

    # ------------------------------------------------------------------
    def _hide(self, map_state, my_pos):
        """Navigate to the cell hardest for pacman to reach from any edge."""
        if self.hide_target is None or map_state[self.hide_target] != 0:
            self.hide_target = self._best_hide_cell(map_state, my_pos)

        if self.hide_target is None or self.hide_target == my_pos:
            return Move.STAY

        move = _astar_first_move(map_state, my_pos, self.hide_target)
        return move if move is not None else Move.STAY

    def _best_hide_cell(self, map_state, my_pos):
        """
        Cell with the highest minimum BFS distance from any map-edge cell.
        Falls back to farthest from last known enemy if no edge cells exist.
        """
        h, w = map_state.shape
        passable = list(map(tuple, np.argwhere(map_state == 0)))
        if not passable:
            return None

        edge_cells = [(r, c) for r, c in passable
                      if r == 0 or r == h-1 or c == 0 or c == w-1]

        if not edge_cells:
            if self.last_enemy is not None:
                d = _bfs_dist(map_state, self.last_enemy)
                return max(passable, key=lambda p: d.get(p, 0))
            return None

        min_dist = {p: _INF for p in passable}
        for ep in edge_cells:
            d = _bfs_dist(map_state, ep)
            for p in passable:
                if d.get(p, _INF) < min_dist[p]:
                    min_dist[p] = d.get(p, _INF)

        reachable = [p for p in passable if min_dist[p] < _INF]
        if not reachable:
            return None
        return max(reachable, key=lambda p: min_dist[p])
