"""
Template for student agent implementation.

INSTRUCTIONS:
1. Copy this file to submissions/<your_student_id>/agent.py
2. Implement the PacmanAgent and/or GhostAgent classes
3. Replace the simple logic with your search algorithm
4. Test your agent using: python arena.py --seek <your_id> --hide example_student

IMPORTANT:
- Do NOT change the class names (PacmanAgent, GhostAgent)
- Do NOT change the method signatures (step, __init__)
- Pacman step must return either a Move or a (Move, steps) tuple where
    1 <= steps <= pacman_speed (provided via kwargs)
- Ghost step must return a Move enum value
- You CAN add your own helper methods
- You CAN import additional Python standard libraries
"""

import sys
from pathlib import Path
from collections import deque
import random

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
# performance benchmark
import time

class MemoryMap:
    def __init__(self):
        self.known_walls = set()       # các ô chắc chắn là tường
        self.known_empty = set()       # các ô chắc chắn là đường đi
        self.last_seen_enemy = None    # vị trí địch lần cuối thấy
        self.step_last_seen = 0        # bước nào thấy lần cuối
        self.unknown_frontier = set()  # biên giới chưa khám phá
        
        # Cache for Topology
        self.topology_cache = {} # Save expensive calculations like static distances
        self.fog_cache = {}      # Lưu mật độ sương mù theo bước

    def update(self, map_state, my_pos, enemy_pos, step_number):
        # Cập nhật bản đồ đã biết
        rows, cols = map_state.shape
        newly_found = set()
        
        # Only check 5x5 area around my_pos
        if map_state[my_pos[0], my_pos[1]] == 0:
            self.known_empty.add(my_pos)
        
        for r in range(max(0, my_pos[0]-5), min(rows, my_pos[0]+6)):
            for c in range(max(0, my_pos[1]-5), min(cols, my_pos[1]+6)):
                if map_state[r][c] == 1:
                    self.known_walls.add((r, c))
                    self.unknown_frontier.discard((r, c))
                elif map_state[r][c] == 0:
                    if (r, c) not in self.known_empty:
                        newly_found.add((r, c))
                    self.known_empty.add((r, c))
                    self.unknown_frontier.discard((r, c))

        # Cập nhật vị trí địch
        if enemy_pos is not None:
            self.last_seen_enemy = enemy_pos
            self.step_last_seen = step_number
            
        # Cập nhật biên giới khám phá (neighbor của known_empty là unknown)
        for pos in newly_found:
            x, y = pos
            for (dx, dy) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if map_state[nx][ny] == -1:  # ô chưa biết
                        self.unknown_frontier.add((nx, ny))
                        
        # Delete cache after increasing step to calculate fog_density more properly
        self.fog_cache.clear()
            
    def get_safe_neighbors(self, pos, map_state):
        """Neighbors that are confirmed empty (==0)."""
        x, y = pos
        rows, cols = map_state.shape
        neighbors = []
        for (dx, dy), move in [
            ((-1,0), Move.UP), ((1,0), Move.DOWN),
            ((0,-1), Move.LEFT), ((0,1), Move.RIGHT)
        ]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_state[nx][ny] == 0:
                    neighbors.append(((nx, ny), move))
        return neighbors

    def get_navigable_neighbors(self, pos, map_state):
        """Neighbors that are NOT confirmed walls (allows fog cells)."""
        x, y = pos
        rows, cols = map_state.shape
        neighbors = []
        for (dx, dy), move in [
            ((-1,0), Move.UP), ((1,0), Move.DOWN),
            ((0,-1), Move.LEFT), ((0,1), Move.RIGHT)
        ]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_state[nx][ny] != 1:  # 0 or -1
                    neighbors.append(((nx, ny), move))
        return neighbors

    def get_exploration_target(self, my_pos, map_state, dist_map=None):
        """Pick closest frontier by BFS distance, with fog-density tiebreak."""
        if not self.unknown_frontier:
            return my_pos

        candidates = list(self.unknown_frontier)
        # Score = -BFS_distance + fog_density_bonus (closer + foggier = better)
        scored = []
        for p in candidates:
            bfs_d = dist_map.get(p, 9999) if dist_map else (abs(p[0]-my_pos[0])+abs(p[1]-my_pos[1]))
            fog_bonus = self.fog_density(p, 3, map_state) * 30
            scored.append((bfs_d - fog_bonus, p))
        scored.sort(key=lambda x: x[0])
        return scored[0][1] if scored else my_pos
    
    def estimate_enemy_pos(self, step_number, map_state):
        """Predict potential enemy locations using BFS with an expanding radius."""
        if self.last_seen_enemy is None:
            return set()
            
        steps_passed = step_number - self.step_last_seen
        # Assume ghost moves 1 cell/step and pacman moves at max_speed. Using a safety radius of steps_passed * 1.2
        radius = int(steps_passed * 1.2) 
        
        possible_positions = {self.last_seen_enemy}
        fringe = deque([(self.last_seen_enemy, 0)])
        visited = {self.last_seen_enemy}
        
        while fringe:
            pos, dist = fringe.popleft()
            if dist >= radius: continue
            
            for (dx, dy) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = pos[0]+dx, pos[1]+dy
                if (nx, ny) not in visited:
                    # Skip known walls; cells marked as -1 or 0 are potential enemy positions
                    if (nx, ny) not in self.known_walls:
                        visited.add((nx, ny))
                        possible_positions.add((nx, ny))
                        fringe.append(((nx, ny), dist + 1))
        return possible_positions
    
    def fog_density(self, center, radius, map_state):
        """Calculate the ratio of unknown cells (-1) within a given radius."""
        cache_key = (center, radius)
        if cache_key in self.fog_cache:
            return self.fog_cache[cache_key]

        r_min, r_max = max(0, center[0]-radius), min(map_state.shape[0], center[0]+radius+1)
        c_min, c_max = max(0, center[1]-radius), min(map_state.shape[1], center[1]+radius+1)
        
        unknown_count = 0
        total_walkable = 0
        
        for r in range(r_min, r_max):
            for c in range(c_min, c_max):
                if (r, c) in self.known_walls: continue
                total_walkable += 1
                if (r, c) not in self.known_empty:
                    unknown_count += 1
        
        density = unknown_count / total_walkable if total_walkable > 0 else 0
        self.fog_cache[cache_key] = density
        return density
    
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        self.memory = MemoryMap()
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Hunter v2 — Speed-BFS + Frontier Sweep"
        self.current_path = []
        self.prev_move = None
        self.visited = set()
        self.last_positions = []
        self.dist_map = {}       # BFS distance (safe cells only)
        self.nav_dist_map = {}   # BFS distance (fog-traversable)
        self.speed_dist_map = {} # speed-aware BFS (game turns)
        self.last_dist_source = None
        self.last_map_hash = None
        # Target commitment
        self.committed_target = None
        self.committed_steps = 0
        self.was_visible = False

    def step(self, map_state, my_position, enemy_position, step_number):
        self.memory.update(map_state, my_position, enemy_position, step_number)
        t0 = time.perf_counter()
        self.visited.add(my_position)

        # ---- CACHE dist_map ----
        current_map_hash = hash(map_state.tobytes())
        if self.last_dist_source != my_position or self.last_map_hash != current_map_hash:
            self.dist_map = self._compute_dist_map(my_position, map_state, fog_ok=False)
            self.nav_dist_map = self._compute_dist_map(my_position, map_state, fog_ok=True)
            self.speed_dist_map = self._compute_speed_dist_map(my_position, map_state)
            self.last_dist_source = my_position
            self.last_map_hash = current_map_hash

        # ---- LOOP DETECTION ----
        self.last_positions.append(my_position)
        if len(self.last_positions) > 8:
            self.last_positions.pop(0)
        is_looping = len(set(self.last_positions)) <= 2 and len(self.last_positions) >= 6

        # ---- Clear visited on loss-of-sight ----
        visible = enemy_position is not None
        if self.was_visible and not visible:
            self.visited.clear()
            self.visited.add(my_position)
        self.was_visible = visible

        # ---- TARGET SELECTION ----
        target = None
        mode = "explore"

        if enemy_position is not None:
            # Direct chase
            target = enemy_position
            mode = "chase"
            self.committed_target = None
            self.committed_steps = 0

        elif self.memory.last_seen_enemy is not None:
            # Predictive intercept
            ghost_start = self.memory.last_seen_enemy
            steps_since = step_number - self.memory.step_last_seen

            if steps_since <= 12:
                ghost_path = self._predict_ghost_path(ghost_start, map_state, steps=min(6, steps_since + 3))
                intercept_points = []
                for t, pos in enumerate(ghost_path, start=1):
                    pac_turns = self.speed_dist_map.get(pos, 9999)
                    if pac_turns <= t:
                        score = pac_turns - t
                        intercept_points.append((pos, score))
                if intercept_points:
                    target = min(intercept_points, key=lambda x: x[1])[0]
                    mode = "intercept"
                else:
                    target = ghost_path[-1]
                    mode = "predict"
            else:
                # Too stale — go to last known, then explore
                target = self.memory.last_seen_enemy
                mode = "stale"
                if my_position == target or self.dist_map.get(target, 9999) <= 1:
                    self.memory.last_seen_enemy = None
                    target = None

        if target is None:
            # Committed exploration target
            if self.committed_target is not None and self.committed_steps < 15:
                if my_position == self.committed_target:
                    self.committed_target = None
                    self.committed_steps = 0
                elif self.committed_target in self.memory.known_empty and \
                     self.committed_target not in self.memory.unknown_frontier:
                    # Frontier was already explored
                    self.committed_target = None
                    self.committed_steps = 0

            if self.committed_target is None or self.committed_steps >= 15 or is_looping:
                self.committed_target = self.memory.get_exploration_target(
                    my_position, map_state, self.nav_dist_map
                )
                self.committed_steps = 0

            target = self.committed_target
            self.committed_steps += 1
            mode = "explore"

        if target is None:
            target = my_position

        # ---- REPLAN ----
        if not self.current_path or is_looping or mode == "chase":
            path = self._bfs_nav(my_position, target, map_state)
            if not path:
                path = self._bfs(my_position, target, map_state)
            if not path:
                frontier_target = self._bfs_to_frontier(my_position, map_state)
                path = self._bfs(my_position, frontier_target, map_state)
            self.current_path = path

        # ---- FALLBACK ----
        if not self.current_path:
            return self._explore(my_position, map_state, True)

        move = self.current_path.pop(0)
        steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
        if steps == 0:
            self.current_path = []
            return self._explore(my_position, map_state, True)

        # ---- MULTI-STEP (speed chaining) ----
        if self.current_path and self.current_path[0] == move and self.pacman_speed >= 2:
            next_pos = self._apply_move(my_position, move)
            next2 = self._apply_move(next_pos, move)
            if self._is_valid_position(next_pos, map_state) and self._is_valid_position(next2, map_state):
                steps = min(2, self.pacman_speed)
                self.current_path.pop(0)
            else:
                steps = 1

        self.prev_move = move
        # print(f"[Pacman] Step {step_number} | mode={mode} target={target} | move={move.name} steps={steps} | Time: {time.perf_counter() - t0:.4f}s")
        return (move, max(1, steps))

    # ------------------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------------------
    def _predict_ghost_move(self, ghost_pos, map_state, prev_pos=None):
        """Predict next Ghost position using heuristic model."""
        neighbors = []
        for (dx, dy), move in [
            ((-1,0), Move.UP), ((1,0), Move.DOWN),
            ((0,-1), Move.LEFT), ((0,1), Move.RIGHT)
        ]:
            nx, ny = ghost_pos[0] + dx, ghost_pos[1] + dy
            if self._is_navigable(nx, ny, map_state):
                neighbors.append(((nx, ny), move))

        if not neighbors:
            return ghost_pos

        best_score = -float('inf')
        best_pos = ghost_pos
        for pos, _ in neighbors:
            dist = self.dist_map.get(pos, 50)
            fog = self.memory.fog_density(pos, 2, map_state)
            deg = len(neighbors)  # lightweight
            # Anti-backtrack: Ghost unlikely to return to previous position
            backtrack = -20 if prev_pos and pos == prev_pos else 0
            score = dist * 12 + fog * 40 + deg * 5 + backtrack
            if score > best_score:
                best_score = score
                best_pos = pos
        return best_pos

    def _predict_ghost_path(self, start_pos, map_state, steps=6):
        """Rollout ghost evasion policy with anti-backtrack."""
        path = []
        current = start_pos
        prev = None
        for _ in range(steps):
            nxt = self._predict_ghost_move(current, map_state, prev)
            path.append(nxt)
            prev = current
            current = nxt
        return path

    # ------------------------------------------------------------------
    # BFS VARIANTS
    # ------------------------------------------------------------------
    def _compute_dist_map(self, start, map_state, fog_ok=False):
        """BFS from start. fog_ok=True allows traversal through fog cells."""
        dist = {start: 0}
        queue = deque([start])
        get_nb = self.memory.get_navigable_neighbors if fog_ok else self.memory.get_safe_neighbors
        while queue:
            current = queue.popleft()
            d = dist[current]
            for next_pos, _ in get_nb(current, map_state):
                if next_pos not in dist:
                    dist[next_pos] = d + 1
                    queue.append(next_pos)
        return dist

    def _compute_speed_dist_map(self, start, map_state):
        """Speed-aware BFS: each step covers 1..speed cells straight.
        Returns {pos: min_game_turns}."""
        dist = {start: 0}
        queue = deque([start])
        while queue:
            p = queue.popleft()
            d = dist[p]
            for mv in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = mv.value
                for n in range(1, self.pacman_speed + 1):
                    nxt = (p[0] + dr * n, p[1] + dc * n)
                    r, c = nxt
                    h, w = map_state.shape
                    if not (0 <= r < h and 0 <= c < w and map_state[r, c] != 1):
                        break
                    if nxt not in dist:
                        dist[nxt] = d + 1
                        queue.append(nxt)
        return dist

    def _bfs(self, start, goal, map_state):
        """BFS on safe cells only."""
        if start == goal:
            return []
        queue = deque([start])
        visited = {start}
        parent = {}
        while queue:
            current = queue.popleft()
            neighbors = self.memory.get_safe_neighbors(current, map_state)
            if current in parent:
                _, prev_move = parent[current]
                neighbors.sort(key=lambda x: x[1] != prev_move)
            for next_pos, move in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)
        return []

    def _bfs_nav(self, start, goal, map_state):
        """BFS allowing fog traversal (navigable neighbors)."""
        if start == goal:
            return []
        queue = deque([start])
        visited = {start}
        parent = {}
        while queue:
            current = queue.popleft()
            neighbors = self.memory.get_navigable_neighbors(current, map_state)
            if current in parent:
                _, prev_move = parent[current]
                neighbors.sort(key=lambda x: x[1] != prev_move)
            for next_pos, move in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)
        return []

    def _bfs_to_frontier(self, start, map_state):
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            if current in self.memory.known_empty:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = current[0]+dx, current[1]+dy
                    if (nx, ny) in self.memory.unknown_frontier:
                        return current
            for next_pos, _ in self.memory.get_safe_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)
        return start

    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal
        while cur != start:
            prev, move = parent[cur]
            path.append(move)
            cur = prev
        return path[::-1]

    def _explore(self, my_position, map_state, force_random=False):
        neighbors = self.memory.get_safe_neighbors(my_position, map_state)
        if not neighbors:
            # Try navigable (fog) neighbors
            neighbors = self.memory.get_navigable_neighbors(my_position, map_state)
        if not neighbors:
            return (Move.STAY, 1)

        if force_random:
            move = random.choice(neighbors)[1]
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            return (move, max(1, steps))

        opposite = {
            Move.UP: Move.DOWN, Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT
        }
        best_move = Move.STAY
        best_score = -float('inf')
        for next_pos, move in neighbors:
            reverse_penalty = -15 if self.prev_move and move == opposite.get(self.prev_move) else 0
            visited_penalty = -5 if next_pos in self.visited else 3
            frontier_bonus = 8 if next_pos in self.memory.unknown_frontier else 0
            fog_bonus = 3 if map_state[next_pos[0], next_pos[1]] == -1 else 0
            score = frontier_bonus + visited_penalty + reverse_penalty + fog_bonus
            if score > best_score:
                best_score = score
                best_move = move

        steps = self._max_valid_steps(my_position, best_move, map_state, self.pacman_speed)
        self.prev_move = best_move
        return (best_move, max(1, steps))

    # ------------------------------------------------------------------
    # HELPERS
    # ------------------------------------------------------------------
    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _is_navigable(self, r, c, map_state):
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps



class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Phantom v11 (Optimized + Particle Filter)
    Improvements over v10:
    - Lightweight Particle Filter (30 particles) for Pacman estimation
    - Cached BFS for Voronoi computation
    - Anti-leapfrog defense
    - Tuned fog exploitation weights
    - Time-bounded computation
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Phantom v11"
        self.pacman_speed = kwargs.get('pacman_speed', 2)
        self.learned_map = None
        self.dead_ends = {}
        self.intersections = set()
        self._topo_hash = None
        self.prev_pos = []
        self.last_seen_pac = None
        self.last_seen_step = 0
        # Particle Filter
        self.particles = []
        self.N_PARTICLES = 30

    def step(self, map_state, my_position, enemy_position, step_number):
        t0 = time.perf_counter()
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
        else:
            vis = map_state >= 0
            self.learned_map[vis] = map_state[vis]

        if enemy_position is not None:
            self.last_seen_pac = enemy_position
            self.last_seen_step = step_number

        self.prev_pos.append(my_position)
        if len(self.prev_pos) > 15:
            self.prev_pos.pop(0)

        th = hash((self.learned_map == 1).tobytes())
        if self._topo_hash != th:
            self._build_topology()
            self._topo_hash = th

        # Particle Filter update
        self._update_particles(enemy_position, my_position, map_state)

        moves = self._legal(my_position, map_state)
        if not moves:
            # print(f"[Ghost ] Step {step_number:3d} | threat={enemy_position or self.last_seen_pac} | move=STAY  | Time: {time.perf_counter() - t0:.4f}s")
            return Move.STAY
        if len(moves) == 1:
            best_mv = moves[0][1]
            # print(f"[Ghost ] Step {step_number:3d} | threat={enemy_position or self.last_seen_pac} | move={best_mv.name:5s} | Time: {time.perf_counter() - t0:.4f}s")
            return best_mv

        # OPENING BOOK: escape ghost house area (rows 7-12 are dangerous corridors)
        in_ghost_house = 7 <= my_position[0] <= 12 and 5 <= my_position[1] <= 15
        pac_ref = enemy_position if enemy_position is not None else self.last_seen_pac
        if pac_ref is not None and (step_number <= 5 or (in_ghost_house and step_number <= 12)):
            best_opening = self._opening_escape(my_position, map_state, moves, pac_ref)
            if best_opening is not None:
                # print(f"[Ghost ] Step {step_number:3d} | threat={enemy_position or self.last_seen_pac} | move={best_opening.name:5s} | Time: {time.perf_counter() - t0:.4f}s")
                return best_opening

        # Determine threat position
        if enemy_position is not None:
            threat = enemy_position
        else:
            threat = self._get_threat_estimate(step_number)

        pac_bfs = self._bfs(threat, map_state) if threat else {}
        my_dist = pac_bfs.get(my_position, 999)

        # Pre-compute ghost BFS for all candidate positions (cached)
        all_candidates = list(set([my_position] + [nxt for nxt, _ in moves]))
        gbfs_cache = {}
        for pos in all_candidates:
            if time.perf_counter() - t0 > 0.6:
                break
            gbfs_cache[pos] = self._bfs(pos, map_state)

        # EMERGENCY MODE: when very close, use fast max-distance logic
        if threat and my_dist <= 3:
            best_mv = self._emergency_escape(my_position, moves, map_state, threat, pac_bfs)
            # print(f"[Ghost ] Step {step_number:3d} | threat={threat} | move={best_mv.name:5s} | Time: {time.perf_counter() - t0:.4f}s")
            return best_mv

        scores = []
        stay_sc = self._score(my_position, Move.STAY, threat, pac_bfs,
                              gbfs_cache.get(my_position), map_state,
                              my_dist, True, my_position, t0)
        scores.append((stay_sc, Move.STAY))

        for nxt, mv in moves:
            if time.perf_counter() - t0 > 0.8:
                d = pac_bfs.get(nxt, 50)
                scores.append((d * 100, mv))
                continue
            sc = self._score(nxt, mv, threat, pac_bfs,
                            gbfs_cache.get(nxt), map_state,
                            my_dist, False, my_position, t0)
            scores.append((sc, mv))

        scores.sort(key=lambda x: x[0], reverse=True)
        best_sc, best_mv = scores[0]

        if best_sc <= -999000:
            best_mv = self._desperate_escape(my_position, moves, map_state, threat)

        # print(f"[Ghost ] Step {step_number:3d} | threat={threat} | move={best_mv.name:5s} | Time: {time.perf_counter() - t0:.4f}s")
        return best_mv

    def _emergency_escape(self, my_pos, moves, ms, pac, pac_bfs):
        """Fast escape when Ghost is critically close to Pacman (dist<=3)."""
        best_mv = moves[0][1] if moves else Move.STAY
        best_score = -float('inf')
        rand = {mv: random.uniform(0, 100) for _, mv in moves}
        for nxt, mv in moves:
            d = pac_bfs.get(nxt, 50)
            score = d * 300  # Maximize BFS distance
            # Multi-step lookahead: how many cells reachable in 5 steps?
            reach, exits = self._escape_plan(nxt, ms, 5)
            score += reach * 50 + exits * 100
            # LoS breaking is critical
            if self._los(nxt, pac, ms):
                score -= 20000
            else:
                score += 5000
            # Dead-end is death
            depth = self.dead_ends.get(nxt, 0)
            if depth > 0:
                score -= depth * 5000
            # Prefer intersections
            if nxt in self.intersections:
                score += 3000
            # Wall-hugging
            score += self._wall_neighbors(nxt, ms) * 100
            # Anti-loop (stronger in emergency)
            score -= self.prev_pos[-8:].count(nxt) * 2000
            # Fog bonus
            if ms[nxt[0], nxt[1]] == -1:
                score += 1500
            # Randomness
            score += rand[mv]
            if score > best_score:
                best_score = score
                best_mv = mv
        return best_mv

    # ------------------------------------------------------------------
    # OPENING BOOK
    # ------------------------------------------------------------------
    def _opening_escape(self, my_pos, map_state, moves, pac_ref=None):
        """Escape phase: break alignment with Pacman position, flee to safety."""
        if pac_ref is None:
            return None  # No Pacman info → skip opening
        pac_spawn = pac_ref

        # Add controlled randomness for unpredictability
        rand_bonus = {mv: random.uniform(0, 150) for _, mv in moves}

        best_mv = None
        best_score = -float('inf')
        for nxt, mv in moves:
            # BFS distance from Pacman spawn to candidate position
            d = abs(nxt[0] - pac_spawn[0]) + abs(nxt[1] - pac_spawn[1])
            exits = len(self._legal(nxt, map_state))
            depth = self.dead_ends.get(nxt, 0)

            score = d * 80 + exits * 50

            # CRITICAL: Must break LoS and column/row alignment
            same_col = (nxt[1] == pac_spawn[1])
            same_row = (nxt[0] == pac_spawn[0])
            in_los = self._los(nxt, pac_spawn, map_state)

            if in_los:
                score -= 5000  # NEVER stay in line of sight
            if same_col:
                score -= 3000  # Break column alignment urgently
            if same_row:
                score -= 1500

            # Reward moves that go lateral (LEFT/RIGHT) when on same column
            if my_pos[1] == pac_spawn[1]:  # Currently on same column
                if mv in (Move.LEFT, Move.RIGHT):
                    score += 3000  # Strongly prefer lateral escape

            # Dead-end avoidance
            if depth > 0:
                score -= depth * 800

            # Prefer intersections
            if nxt in self.intersections:
                score += 600

            # Avoid long corridors
            stretch = self._open_stretch(nxt, map_state)
            if stretch >= 4:
                score -= stretch * 50

            # Randomness for unpredictability
            score += rand_bonus[mv]

            if score > best_score:
                best_score = score
                best_mv = mv
        return best_mv

    # ------------------------------------------------------------------
    # PARTICLE FILTER
    # ------------------------------------------------------------------
    def _update_particles(self, enemy_position, my_position, map_state):
        h, w = map_state.shape
        if enemy_position is not None:
            self.particles = [enemy_position] * self.N_PARTICLES
            return
        if not self.particles:
            candidates = []
            for r in range(h):
                for c in range(w):
                    if map_state[r, c] != 1:
                        candidates.append((r, c))
            if candidates:
                self.particles = [random.choice(candidates)
                                  for _ in range(self.N_PARTICLES)]
            return

        new_particles = []
        for pos in self.particles:
            legal = self._legal(pos, map_state)
            if not legal:
                new_particles.append(pos)
                continue
            if random.random() < 0.7 and my_position:
                best_pos = pos
                best_d = abs(pos[0] - my_position[0]) + abs(pos[1] - my_position[1])
                for nxt, mv in legal:
                    d = abs(nxt[0] - my_position[0]) + abs(nxt[1] - my_position[1])
                    if d < best_d:
                        best_d, best_pos = d, nxt
                    # Pacman speed-2: check 2nd step in same direction
                    nxt2 = (nxt[0] + mv.value[0], nxt[1] + mv.value[1])
                    if (0 <= nxt2[0] < h and 0 <= nxt2[1] < w
                            and map_state[nxt2[0], nxt2[1]] != 1):
                        d2 = abs(nxt2[0] - my_position[0]) + abs(nxt2[1] - my_position[1])
                        if d2 < best_d:
                            best_d, best_pos = d2, nxt2
                pos = best_pos
            else:
                pos = random.choice(legal)[0]
            # Reject particles in visible cells (Pacman would be seen there)
            if (0 <= pos[0] < h and 0 <= pos[1] < w
                    and map_state[pos[0], pos[1]] != -1):
                pos = random.choice(self.particles)
            new_particles.append(pos)
        self.particles = new_particles

    def _get_threat_estimate(self, step_number):
        """Best estimate of Pacman position from particles or memory."""
        if self.particles:
            counts = {}
            for p in self.particles:
                counts[p] = counts.get(p, 0) + 1
            return max(counts, key=counts.get)
        if self.last_seen_pac is not None:
            max_age = max(20, int(15 * self.pacman_speed))
            if step_number - self.last_seen_step <= max_age:
                return self.last_seen_pac
        return None

    # ------------------------------------------------------------------
    # SCORING
    # ------------------------------------------------------------------
    def _score(self, nxt, move_dir, pac, pbfs, gbfs, ms, current_dist,
               is_stay, my_pos, t0):
        s = 0.0
        dist = pbfs.get(nxt, 50)
        turns_to_pac = dist / max(1, self.pacman_speed)

        # M1: TOPOLOGY ESCAPE
        reach, exits = self._escape_plan(nxt, ms, 5)
        s += reach * 15 + exits * 40
        if exits == 0 and turns_to_pac < 15:
            s -= 10000

        # M2: VORONOI TERRITORY (uses cached gbfs)
        if gbfs is not None and time.perf_counter() - t0 < 0.7:
            territory, vor_exits = self._voronoi_territory(
                nxt, pac, pbfs, ms, gbfs)
        else:
            territory, vor_exits = 200, 0
        if pac and territory < 20:
            s -= (40 - territory) * 800
        s += territory * 60

        # M3: DISTANCE FROM PACMAN (tuned for capture_distance=2, speed=2)
        s += turns_to_pac * 200
        if dist < current_dist and current_dist <= 25:
            s -= 8000
        elif dist > current_dist:
            s += 800
        # Graduated panic: the closer, the worse
        if pac:
            if turns_to_pac < 1.0:
                return -999999  # Immediate death
            elif turns_to_pac < 1.5:
                s -= 50000  # Critical danger
            elif turns_to_pac < 2.5:
                s -= 15000  # High danger
            elif turns_to_pac < 4.0:
                s -= 3000   # Elevated danger

        # M4: DEAD-END RISK
        depth = self.dead_ends.get(nxt, 0)
        if depth > 0:
            if turns_to_pac < 8:
                s -= (depth ** 2) * 1200
            else:
                s -= depth * 150

        # M5: LINE OF SIGHT BREAKING (strengthened)
        if pac and self._los(nxt, pac, ms):
            los_dist = abs(nxt[0] - pac[0]) + abs(nxt[1] - pac[1])
            los_turns = los_dist / max(1, self.pacman_speed)
            if los_turns <= 1:
                s -= 25000  # Almost certain death
            elif los_turns <= 2:
                s -= 12000  # Very dangerous
            elif los_turns <= 3:
                s -= 5000
            else:
                s -= 1500
        elif pac:
            s += 2000  # Strong reward for hiding

        # M6: ORTHOGONAL JUKING
        if pac and not is_stay:
            vx = nxt[0] - pac[0]
            vy = nxt[1] - pac[1]
            mx = nxt[0] - my_pos[0]
            my = nxt[1] - my_pos[1]
            dot = (vx * mx) + (vy * my)
            if dot == 0:
                s += 600
                if self._wall_neighbors(nxt, ms) >= 2:
                    s += 300

        # M7: FOG DENSITY & SPRINT CORRIDORS (Tuned weights)
        fog = self._fog7(nxt, ms)
        if is_stay:
            s += fog * (40 if turns_to_pac > 10 else 5)
        else:
            s += fog * 25  # Increased from 15 to encourage active fog-seeking
        if ms[nxt[0], nxt[1]] == -1:
            s += 200

        open_stretch = self._open_stretch(nxt, ms)
        wall_count = self._wall_neighbors(nxt, ms)

        if open_stretch >= 4:
            s -= open_stretch * 60
        s += wall_count * 20

        if nxt in self.intersections:
            s += 200
            if turns_to_pac < 8:
                s += 400

        # M8: ANTI-LOOP
        if not is_stay:
            s -= self.prev_pos[-6:].count(nxt) * 500

        # M9: DYNAMIC STAY DECISION
        if is_stay:
            s -= 1000
            recent_stays = sum(1 for i in range(1, len(self.prev_pos))
                               if self.prev_pos[i] == self.prev_pos[i - 1])
            if recent_stays >= 3:
                s -= 80000
            if (fog >= 12 and depth == 0 and turns_to_pac > 10
                    and open_stretch <= 2 and exits > 0):
                s += 2500
            else:
                s -= 40000

        # M10: ANTI-LEAPFROG DEFENSE (NEW)
        if pac and not is_stay:
            if self._leapfrog_risk(nxt, pac, ms):
                s -= 8000

        return s

    # ------------------------------------------------------------------
    # ANTI-LEAPFROG
    # ------------------------------------------------------------------
    def _leapfrog_risk(self, ghost_pos, pac_pos, ms):
        """Detect if Pacman can jump past Ghost in a straight corridor."""
        if pac_pos is None:
            return False
        dr = ghost_pos[0] - pac_pos[0]
        dc = ghost_pos[1] - pac_pos[1]
        total = abs(dr) + abs(dc)
        # Only same axis and within leapfrog range
        if dr != 0 and dc != 0:
            return False
        if total > self.pacman_speed + 1 or total == 0:
            return False
        if not self._los(ghost_pos, pac_pos, ms):
            return False
        # Check landing cell behind Ghost
        if dr == 0:
            step_c = 1 if dc > 0 else -1
            behind = (ghost_pos[0], ghost_pos[1] + step_c)
        else:
            step_r = 1 if dr > 0 else -1
            behind = (ghost_pos[0] + step_r, ghost_pos[1])
        h, w = ms.shape
        return (0 <= behind[0] < h and 0 <= behind[1] < w
                and ms[behind[0], behind[1]] != 1)

    # ------------------------------------------------------------------
    # VORONOI TERRITORY (accepts pre-computed gbfs)
    # ------------------------------------------------------------------
    def _voronoi_territory(self, g_pos, p_pos, pbfs, ms, gbfs=None):
        """Ghost territory = cells Ghost reaches before Pacman (speed-aware)."""
        if p_pos is None:
            return 200, 0
        if gbfs is None:
            gbfs = self._bfs(g_pos, ms)
        territory = 0
        exits = 0
        h, w = ms.shape
        for r in range(h):
            for c in range(w):
                if ms[r, c] == 1:
                    continue
                node = (r, c)
                g_turns = gbfs.get(node, 999)
                p_turns = pbfs.get(node, 999) / max(1, self.pacman_speed)
                if g_turns < p_turns:
                    territory += 1
                    if g_turns > 5:
                        exits += 1
        return territory, exits

    def _desperate_escape(self, pos, moves, ms, pac):
        best_mv = moves[0][1] if moves else Move.STAY
        best_reach = -1
        for nxt, mv in moves:
            reach, exits = self._escape_plan(nxt, ms, 5)
            away = 0
            if pac:
                d_cur = abs(pos[0] - pac[0]) + abs(pos[1] - pac[1])
                d_new = abs(nxt[0] - pac[0]) + abs(nxt[1] - pac[1])
                away = d_new - d_cur
            score = reach * 15 + exits * 50 + away * 10
            if pac and self._los(nxt, pac, ms):
                score -= 500
            if score > best_reach:
                best_reach = score
                best_mv = mv
        return best_mv

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------
    def _escape_plan(self, pos, ms, depth):
        visited = {pos}
        frontier = [pos]
        h, w = ms.shape
        for _ in range(depth):
            nxt_frontier = []
            for cur in frontier:
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nx, ny = cur[0] + dx, cur[1] + dy
                    n = (nx, ny)
                    if (0 <= nx < h and 0 <= ny < w
                            and n not in visited and ms[nx, ny] != 1):
                        visited.add(n)
                        nxt_frontier.append(n)
            frontier = nxt_frontier
        return len(visited) - 1, len(frontier)

    def _legal(self, pos, ms):
        x, y = pos
        h, w = ms.shape
        out = []
        for (dx, dy), mv in [((-1, 0), Move.UP), ((1, 0), Move.DOWN),
                              ((0, -1), Move.LEFT), ((0, 1), Move.RIGHT)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and ms[nx, ny] != 1:
                out.append(((nx, ny), mv))
        return out

    def _bfs(self, start, ms):
        if start is None:
            return {}
        dist = {start: 0}
        q = deque([start])
        h, w = ms.shape
        while q:
            p = q.popleft()
            d = dist[p]
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = p[0] + dx, p[1] + dy
                n = (nx, ny)
                if (0 <= nx < h and 0 <= ny < w
                        and n not in dist and ms[nx, ny] != 1):
                    dist[n] = d + 1
                    q.append(n)
        return dist

    def _los(self, a, b, ms):
        if b is None:
            return False
        ar, ac = a
        br, bc = b
        if ar == br:
            lo, hi = min(ac, bc), max(ac, bc)
            for c in range(lo, hi + 1):
                if ms[ar, c] == 1:
                    return False
            return True
        if ac == bc:
            lo, hi = min(ar, br), max(ar, br)
            for r in range(lo, hi + 1):
                if ms[r, ac] == 1:
                    return False
            return True
        return False

    def _fog7(self, pos, ms):
        x, y = pos
        h, w = ms.shape
        c = 0
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                nx, ny = x + dx, y + dy
                if 0 <= nx < h and 0 <= ny < w and ms[nx, ny] == -1:
                    c += 1
        return c

    def _wall_neighbors(self, pos, ms):
        x, y = pos
        h, w = ms.shape
        c = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if not (0 <= nx < h and 0 <= ny < w) or ms[nx, ny] == 1:
                    c += 1
        return c

    def _open_stretch(self, pos, ms):
        h, w = ms.shape
        best = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            count = 0
            r, c = pos
            for _ in range(8):
                r, c = r + dr, c + dc
                if 0 <= r < h and 0 <= c < w and ms[r, c] != 1:
                    count += 1
                else:
                    break
            best = max(best, count)
        return best

    def _build_topology(self):
        ms = self.learned_map
        h, w = ms.shape
        deg = {}
        self.intersections.clear()
        self.dead_ends.clear()
        for r in range(h):
            for c in range(w):
                if ms[r, c] != 1:
                    d = sum(1 for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]
                            if 0 <= r + dx < h and 0 <= c + dy < w
                            and ms[r + dx, c + dy] != 1)
                    deg[(r, c)] = d
                    if d >= 3:
                        self.intersections.add((r, c))
        dead = {p: 1 for p, d in deg.items() if d <= 1}
        q = deque(dead.keys())
        self.dead_ends = dead.copy()
        while q:
            cur = q.popleft()
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nx, ny = cur[0] + dx, cur[1] + dy
                nxt = (nx, ny)
                if deg.get(nxt, 0) == 2 and nxt not in self.dead_ends:
                    self.dead_ends[nxt] = self.dead_ends[cur] + 1
                    q.append(nxt)
