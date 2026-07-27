import sys, importlib.util, heapq, urllib.request
from pathlib import Path
from collections import deque

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

_CACHE_DIR  = Path(__file__).parent / '_rl_cache'
_RAW_SCRIPT = "https://raw.githubusercontent.com/schizocatto/RL-hideseek/main/rl_agent.py"
_HIST_LEN   = 4

def _load_rl_module():
    _CACHE_DIR.mkdir(exist_ok=True)
    script = _CACHE_DIR / 'rl_agent.py'
    if not script.exists():
        urllib.request.urlretrieve(_RAW_SCRIPT, str(script))
    spec = importlib.util.spec_from_file_location('_rl_agent', script)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_RL = None
try:
    _RL = _load_rl_module()
    print("[RL] script loaded")
except Exception as e:
    print(f"[RL] heuristic fallback ({e})")

_DIRS       = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
_GHOST_DIRS = [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]
_INF_DIST   = 441

def _valid(pos, map_state):
    r, c = pos
    h, w = map_state.shape
    return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

def _bfs_dist(map_state, start, dirs=None):
    if dirs is None: dirs = _DIRS
    dist = {start: 0}
    q = deque([start])
    while q:
        p = q.popleft(); d = dist[p]
        for mv in dirs:
            dr, dc = mv.value
            nxt = (p[0]+dr, p[1]+dc)
            if nxt not in dist and _valid(nxt, map_state):
                dist[nxt] = d+1; q.append(nxt)
    return dist

def _bfs_speed_dist(map_state, start, max_speed):
    dist = {start: 0}
    q = deque([start])
    while q:
        p = q.popleft(); d = dist[p]
        for mv in _DIRS:
            dr, dc = mv.value
            for n in range(1, max_speed+1):
                nxt = (p[0]+dr*n, p[1]+dc*n)
                if not _valid(nxt, map_state): break
                if nxt not in dist:
                    dist[nxt] = d+1; q.append(nxt)
    return dist

def _astar(map_state, start, goal):
    if start == goal: return [start]
    h = lambda p: abs(p[0]-goal[0])+abs(p[1]-goal[1])
    heap = [(h(start), 0, start)]
    came_from = {}; g = {start: 0}
    while heap:
        _, cost, p = heapq.heappop(heap)
        if p == goal:
            path = [p]
            while p in came_from: p = came_from[p]; path.append(p)
            return list(reversed(path))
        if cost > g.get(p, _INF_DIST): continue
        for mv in _DIRS:
            dr, dc = mv.value
            nxt = (p[0]+dr, p[1]+dc)
            if not _valid(nxt, map_state): continue
            nc = cost+1
            if nc < g.get(nxt, _INF_DIST):
                g[nxt] = nc; came_from[nxt] = p
                heapq.heappush(heap, (nc+h(nxt), nc, nxt))
    return None

def _astar_speed(map_state, start, goal, max_speed):
    if start == goal: return Move.STAY, 1
    h = lambda p: (abs(p[0]-goal[0])+abs(p[1]-goal[1]))/max_speed
    heap = [(h(start), 0, start)]
    came_from = {}; g = {start: 0}
    while heap:
        _, cost, p = heapq.heappop(heap)
        if p == goal:
            cur = p
            while came_from[cur][0] != start: cur = came_from[cur][0]
            _, first_move, first_steps = came_from[cur]
            return first_move, first_steps
        if cost > g.get(p, _INF_DIST): continue
        for mv in _DIRS:
            dr, dc = mv.value
            for n in range(1, max_speed+1):
                nxt = (p[0]+dr*n, p[1]+dc*n)
                if not _valid(nxt, map_state): break
                nc = cost+1
                if nc < g.get(nxt, _INF_DIST):
                    g[nxt] = nc; came_from[nxt] = (p, mv, n)
                    heapq.heappush(heap, (nc+h(nxt), nc, nxt))
    return Move.STAY, 1

def _to_move(from_pos, to_pos):
    dr, dc = to_pos[0]-from_pos[0], to_pos[1]-from_pos[1]
    for mv in Move:
        if mv.value == (dr, dc): return mv
    return Move.STAY

def _exits(pos, map_state):
    return sum(1 for mv in _DIRS if _valid((pos[0]+mv.value[0], pos[1]+mv.value[1]), map_state))

def _safe_zone_components(safe_zone_set):
    comp_size = {}; visited = set()
    for start in safe_zone_set:
        if start in visited: continue
        component = []; q = deque([start]); visited.add(start)
        while q:
            p = q.popleft(); component.append(p)
            for mv in _DIRS:
                nxt = (p[0]+mv.value[0], p[1]+mv.value[1])
                if nxt in safe_zone_set and nxt not in visited:
                    visited.add(nxt); q.append(nxt)
        size = len(component)
        for p in component: comp_size[p] = size
    return comp_size

def _pacman_intercept_move(map_state, my_pos, enemy_pos, max_speed):
    ghost_dist = _bfs_dist(map_state, enemy_pos)
    best_move, best_steps, best_safe, best_tie = Move.STAY, 1, float('inf'), float('inf')
    for mv in _DIRS:
        dr, dc = mv.value
        for n in range(1, max_speed+1):
            nxt = (my_pos[0]+dr*n, my_pos[1]+dc*n)
            if not _valid(nxt, map_state): break
            new_pac   = _bfs_speed_dist(map_state, nxt, max_speed)
            safe_size = sum(1 for pos, gd in ghost_dist.items() if gd <= new_pac.get(pos, _INF_DIST))
            tie = abs(nxt[0]-enemy_pos[0])+abs(nxt[1]-enemy_pos[1])
            if safe_size < best_safe or (safe_size == best_safe and tie < best_tie):
                best_safe, best_tie, best_move, best_steps = safe_size, tie, mv, n
    return best_move, best_steps


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed     = max(1, int(kwargs.get("pacman_speed", 1)))
        self.known_map        = None
        self.last_known_enemy = None
        self.steps_since_seen = 0
        self._visit_counts    = {}
        self._pos_history     = deque(maxlen=_HIST_LEN)

    def step(self, map_state, my_position, enemy_position, step_number):
        if _RL is not None:
            try:
                action = _RL.seeker_move(
                    map_state, my_position, enemy_position,
                    step_number, self._pos_history, self.pacman_speed)
                self._pos_history.append(my_position)
                return action
            except Exception:
                pass
        return self._heuristic_step(map_state, my_position, enemy_position, step_number)

    def _update_state(self, map_state, enemy_pos):
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            mask = map_state != -1
            self.known_map[mask] = map_state[mask]
        if enemy_pos is not None:
            self.last_known_enemy = enemy_pos; self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

    def _nearest_frontier(self, my_pos):
        if self.known_map is None: return None
        h, w = self.known_map.shape
        q = deque([my_pos]); visited = {my_pos}
        while q:
            p = q.popleft(); r, c = p
            for mv in _DIRS:
                nr, nc = r+mv.value[0], c+mv.value[1]
                if 0 <= nr < h and 0 <= nc < w and self.known_map[nr, nc] == -1:
                    return p
            for mv in _DIRS:
                nr, nc = r+mv.value[0], c+mv.value[1]
                nxt = (nr, nc)
                if nxt not in visited and 0 <= nr < h and 0 <= nc < w and self.known_map[nr, nc] == 0:
                    visited.add(nxt); q.append(nxt)
        return None

    def _least_visited_move(self, map_state, my_pos):
        best_mv, best_steps, best_count = Move.STAY, 1, float('inf')
        for mv in _DIRS:
            dr, dc = mv.value
            for n in range(1, self.pacman_speed+1):
                nxt = (my_pos[0]+dr*n, my_pos[1]+dc*n)
                if not _valid(nxt, map_state): break
                cnt = self._visit_counts.get(nxt, 0)
                if cnt < best_count:
                    best_count, best_mv, best_steps = cnt, mv, n
        return best_mv, best_steps

    def _heuristic_step(self, map_state, my_pos, enemy_pos, step_number):
        self._update_state(map_state, enemy_pos)
        self._visit_counts[my_pos] = self._visit_counts.get(my_pos, 0) + 1

        if enemy_pos is None:
            if self.last_known_enemy is not None and self.steps_since_seen < 15:
                move, steps = _astar_speed(map_state, my_pos, self.last_known_enemy, self.pacman_speed)
                if move != Move.STAY: return move, steps
            frontier = self._nearest_frontier(my_pos)
            if frontier is not None and frontier != my_pos:
                move, steps = _astar_speed(map_state, my_pos, frontier, self.pacman_speed)
                if move != Move.STAY: return move, steps
            return self._least_visited_move(map_state, my_pos)

        move, steps = _pacman_intercept_move(map_state, my_pos, enemy_pos, self.pacman_speed)
        if move != Move.STAY:
            dr, dc = move.value
            er, ec = enemy_pos[0]-my_pos[0], enemy_pos[1]-my_pos[1]
            same_axis = (dc==0 and ec==0 and er*dr>0) or (dr==0 and er==0 and ec*dc>0)
            if same_axis:
                gap = abs(er)+abs(ec)
                leapfrog = gap+1
                if leapfrog <= self.pacman_speed:
                    nxt = (my_pos[0]+dr*leapfrog, my_pos[1]+dc*leapfrog)
                    if _valid(nxt, map_state): steps = leapfrog
        return move, steps


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.speed_factor     = max(1, int(kwargs.get("pacman_speed", 2)))
        self.known_map        = None
        self.last_known_enemy = None
        self.steps_since_seen = 0
        self._visit_counts    = {}
        self._pos_history     = deque(maxlen=_HIST_LEN)

    def step(self, map_state, my_position, enemy_position, step_number):
        if _RL is not None:
            try:
                move = _RL.hider_move(
                    map_state, my_position, enemy_position,
                    step_number, self._pos_history)
                self._pos_history.append(my_position)
                return move
            except Exception:
                pass
        return self._heuristic_step(map_state, my_position, enemy_position, step_number)

    def _update_state(self, map_state, enemy_pos):
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            mask = map_state != -1
            self.known_map[mask] = map_state[mask]
        if enemy_pos is not None:
            self.last_known_enemy = enemy_pos; self.steps_since_seen = 0
        else:
            self.steps_since_seen += 1

    def _branching_walk(self, map_state, my_pos):
        best_mv, best_score = Move.STAY, -float('inf')
        for mv in _GHOST_DIRS:
            dr, dc = mv.value
            nxt = (my_pos[0]+dr, my_pos[1]+dc)
            if not _valid(nxt, map_state): continue
            score = _exits(nxt, map_state)*10 - self._visit_counts.get(nxt, 0)*30
            if score > best_score: best_score, best_mv = score, mv
        return best_mv

    def _greedy_evade(self, map_state, my_pos, pacman_dist):
        best_mv    = Move.STAY
        best_score = pacman_dist.get(my_pos, 0)*100 + _exits(my_pos, map_state)*10
        for mv in _GHOST_DIRS:
            dr, dc = mv.value
            nxt = (my_pos[0]+dr, my_pos[1]+dc)
            if not _valid(nxt, map_state): continue
            score = (pacman_dist.get(nxt, 0)*100 + _exits(nxt, map_state)*10
                     - self._visit_counts.get(nxt, 0)*30)
            if score > best_score: best_score, best_mv = score, mv
        return best_mv

    def _heuristic_step(self, map_state, my_pos, enemy_pos, step_number):
        self._update_state(map_state, enemy_pos)
        self._visit_counts[my_pos] = self._visit_counts.get(my_pos, 0) + 1

        if enemy_pos is None:
            if self.last_known_enemy is not None and self.steps_since_seen < 15:
                stale = _bfs_speed_dist(map_state, self.last_known_enemy, self.speed_factor)
                return self._greedy_evade(map_state, my_pos, stale)
            return self._branching_walk(map_state, my_pos)

        pacman_steps = _bfs_speed_dist(map_state, enemy_pos, self.speed_factor)
        ghost_dist   = _bfs_dist(map_state, my_pos, dirs=_GHOST_DIRS)
        safe_zone    = {pos for pos, gd in ghost_dist.items()
                        if gd > 0 and gd <= pacman_steps.get(pos, _INF_DIST)}
        comp_sizes   = _safe_zone_components(safe_zone)

        best_target, best_score = None, -1
        for pos in safe_zone:
            ps     = pacman_steps.get(pos, _INF_DIST)
            comp   = comp_sizes.get(pos, 1)
            visits = self._visit_counts.get(pos, 0)
            score  = ps*100 + comp*5 + _exits(pos, map_state)*10 - visits*50
            if score > best_score: best_score, best_target = score, pos

        if best_target is not None:
            path = _astar(map_state, my_pos, best_target)
            if path and len(path) >= 2:
                return _to_move(my_pos, path[1])

        return self._greedy_evade(map_state, my_pos, pacman_steps)
