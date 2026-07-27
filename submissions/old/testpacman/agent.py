"""
TestPacman - Advanced Pacman (Seeker) Agent

Strategies:
1. Speed-aware A* pathfinding that accounts for multi-tile movement
2. Ghost position prediction based on movement patterns
3. Smart patrol: BFS toward oldest-seen frontier cells, spiraling inward
4. Intercept logic: predict ghost escape direction and cut off paths
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
from collections import deque
from heapq import heappush, heappop


class PacmanAgent(BasePacmanAgent):
    """
    Interceptor Pacman — speed-aware A* with ghost trajectory prediction
    and aggressive fog-clearing patrol.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Interceptor Pacman"
        self.learned_map = None
        self.visit_map = None
        self.last_known_enemy_pos = None
        self.enemy_history = []          # last N positions of the ghost
        self.predicted_ghost_pos = None
        self.step_count = 0

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        self.step_count = step_number

        # --- 1. Build / update memory map ---
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
            self.visit_map = np.zeros_like(map_state, dtype=int)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]

        visible = map_state != -1
        self.visit_map[visible] = step_number

        # --- 2. Ghost tracking & prediction ---
        if enemy_position is not None:
            self.enemy_history.append(enemy_position)
            if len(self.enemy_history) > 8:
                self.enemy_history.pop(0)
            self.last_known_enemy_pos = enemy_position
            self.predicted_ghost_pos = self._predict_ghost(enemy_position)
        else:
            # Ghost not visible — age out prediction after a while
            if self.last_known_enemy_pos is not None:
                # Try to predict where ghost went
                self.predicted_ghost_pos = self._predict_ghost(self.last_known_enemy_pos)

        # --- 3. Choose target ---
        target = None
        if enemy_position is not None:
            # Ghost is visible — go straight for it
            target = enemy_position
        elif self.predicted_ghost_pos is not None:
            target = self.predicted_ghost_pos
        elif self.last_known_enemy_pos is not None:
            target = self.last_known_enemy_pos

        # Arrived at target but ghost not there — clear target
        if target is not None and target == my_position and enemy_position is None:
            self.last_known_enemy_pos = None
            self.predicted_ghost_pos = None
            target = None

        # --- 4. Opening rush: go RIGHT then UP to reach ghost spawn area ---
        # Ghost spawns in upper half. Go right first (ghost 11 goes right),
        # then rush upward to sweep upper corridors
        if target is None and step_number <= 2:
            h, w = self.learned_map.shape
            # Step 1: go RIGHT (toward right corridors)
            if step_number == 1:
                if self._max_valid_steps(my_position, Move.RIGHT, 1) > 0:
                    return (Move.RIGHT, 1)
            # Step 2: go UP aggressively
            if step_number == 2:
                up_steps = self._max_valid_steps(my_position, Move.UP, self.pacman_speed)
                if up_steps > 0:
                    return (Move.UP, up_steps)

        # --- 5. Pathfinding to target ---
        if target is not None:
            path = self._speed_aware_astar(my_position, target)
            if path and path[0] != Move.STAY:
                return self._execute_sprint(my_position, path)
            # A* failed (blocked) — fallback to patrol
            # But first try intercept points if ghost was recently seen
            if enemy_position is not None or (self.last_known_enemy_pos is not None and 
                                                step_number - self._last_seen_step() < 5):
                intercept = self._find_intercept_target(my_position)
                if intercept is not None:
                    path = self._speed_aware_astar(my_position, intercept)
                    if path and path[0] != Move.STAY:
                        return self._execute_sprint(my_position, path)

        # --- 6. Smart patrol to find ghost ---
        path = self._smart_patrol(my_position, step_number)
        if path and path[0] != Move.STAY:
            return self._execute_sprint(my_position, path)

        return (Move.STAY, 1)

    # ─────────────── Ghost prediction ───────────────
    def _predict_ghost(self, last_pos):
        """Predict where ghost is heading based on movement history."""
        if len(self.enemy_history) < 2:
            return last_pos

        # Compute average velocity from recent history
        dr, dc = 0, 0
        count = 0
        for i in range(1, len(self.enemy_history)):
            prev = self.enemy_history[i - 1]
            curr = self.enemy_history[i]
            dr += curr[0] - prev[0]
            dc += curr[1] - prev[1]
            count += 1

        if count == 0:
            return last_pos

        avg_dr = dr / count
        avg_dc = dc / count

        # Project 2-3 steps ahead
        pred_r = int(round(last_pos[0] + avg_dr * 2))
        pred_c = int(round(last_pos[1] + avg_dc * 2))

        # Clamp to map bounds and check walkability
        h, w = self.learned_map.shape
        pred_r = max(0, min(pred_r, h - 1))
        pred_c = max(0, min(pred_c, w - 1))

        if self.learned_map[pred_r, pred_c] == 0:
            return (pred_r, pred_c)

        # If predicted position is a wall, find nearest walkable cell
        best = last_pos
        best_dist = float('inf')
        for dr2 in range(-2, 3):
            for dc2 in range(-2, 3):
                nr, nc = pred_r + dr2, pred_c + dc2
                if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                    d = abs(nr - pred_r) + abs(nc - pred_c)
                    if d < best_dist:
                        best_dist = d
                        best = (nr, nc)
        return best

    def _last_seen_step(self):
        """Return the step at which we last saw the ghost (approximately)."""
        if self.last_known_enemy_pos is None:
            return 0
        r, c = self.last_known_enemy_pos
        return self.visit_map[r, c]

    def _find_intercept_target(self, my_position):
        """Try to find a position that cuts off likely ghost escape routes."""
        if not self.enemy_history:
            return None

        ghost_pos = self.enemy_history[-1]

        # Find all junction cells (cells with 3+ open neighbors) near ghost
        h, w = self.learned_map.shape
        junctions = []
        search_radius = 6

        for r in range(max(0, ghost_pos[0] - search_radius),
                       min(h, ghost_pos[0] + search_radius + 1)):
            for c in range(max(0, ghost_pos[1] - search_radius),
                           min(w, ghost_pos[1] + search_radius + 1)):
                if self.learned_map[r, c] != 0:
                    continue
                # Count open neighbors
                neighbors = 0
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = r + move.value[0], c + move.value[1]
                    if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                        neighbors += 1
                if neighbors >= 3:
                    junctions.append((r, c))

        if not junctions:
            return None

        # Pick junction that is:
        # - Between us and the ghost (on the ghost's likely escape side)
        # - Closest to where we can reach quickly
        ghost_r, ghost_c = ghost_pos
        my_r, my_c = my_position

        best_junction = None
        best_score = float('inf')

        for jr, jc in junctions:
            # Distance from us to junction (cost for us)
            our_dist = abs(jr - my_r) + abs(jc - my_c)
            # Distance from ghost to junction
            ghost_dist = abs(jr - ghost_r) + abs(jc - ghost_c)

            # We want junctions that we can reach before or around the same
            # time as the ghost, accounting for our speed advantage
            our_turns = our_dist / self.pacman_speed
            ghost_turns = ghost_dist  # ghost speed = 1

            # Prefer junctions where we arrive at same time or sooner than ghost
            if our_turns <= ghost_turns + 1:
                score = our_dist  # pick closest reachable intercept
                if score < best_score:
                    best_score = score
                    best_junction = (jr, jc)

        return best_junction

    # ─────────────── Speed-aware A* ───────────────
    def _speed_aware_astar(self, start, goal):
        """A* that accounts for pacman speed: moving N tiles along a straight
        corridor costs only 1 step (turn), not N."""

        def heuristic(pos):
            md = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            # Each turn we can cover up to pacman_speed tiles
            return md / self.pacman_speed

        frontier = [(heuristic(start), 0, start, [])]  # (f, g, pos, path)
        visited = {}  # pos -> best g

        while frontier:
            f_cost, g_cost, current, path = heappop(frontier)

            if current == goal:
                return path

            if current in visited and visited[current] <= g_cost:
                continue
            visited[current] = g_cost

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                # Try all possible step counts (1..pacman_speed) for this move
                pos = current
                moves_for_this_dir = []
                for step_count in range(1, self.pacman_speed + 1):
                    nr = pos[0] + move.value[0]
                    nc = pos[1] + move.value[1]
                    h, w = self.learned_map.shape
                    if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                        pos = (nr, nc)
                        moves_for_this_dir.append(move)
                    else:
                        break

                    # Each step_count in the same direction = 1 turn
                    new_g = g_cost + 1
                    if pos not in visited or visited[pos] > new_g:
                        new_path = path + moves_for_this_dir
                        new_f = new_g + heuristic(pos)
                        heappush(frontier, (new_f, new_g, pos, new_path))

        return [Move.STAY]

    # ─────────────── Sprint execution ───────────────
    def _execute_sprint(self, my_position, path):
        """Given a path of Move objects, bundle consecutive same-direction
        moves into a single (Move, steps) tuple, respecting pacman_speed."""
        if not path or path[0] == Move.STAY:
            return (Move.STAY, 1)

        first_move = path[0]
        desired_steps = 1
        for m in path[1:self.pacman_speed]:
            if m == first_move:
                desired_steps += 1
            else:
                break

        actual = self._max_valid_steps(my_position, first_move, desired_steps)
        if actual > 0:
            return (first_move, actual)
        return (Move.STAY, 1)

    def _max_valid_steps(self, pos, move, max_steps):
        steps = 0
        current = pos
        for _ in range(max_steps):
            nr = current[0] + move.value[0]
            nc = current[1] + move.value[1]
            h, w = self.learned_map.shape
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                steps += 1
                current = (nr, nc)
            else:
                break
        return steps

    # ─────────────── Smart patrol ───────────────
    def _smart_patrol(self, my_position, step_number):
        """Aggressive patrol: rush toward the upper half of the map (ghost spawn area),
        then sweep across. Uses weighted BFS where upward cells get priority."""

        h, w = self.learned_map.shape

        # Strategy: find the best frontier/stale target, weighted by:
        # 1. Being in the upper half (ghost spawn area)
        # 2. Staleness (not visited recently)
        # 3. Proximity to fog

        queue = deque([(my_position, [])])
        visited = {my_position}

        candidates = []  # (score, path)

        while queue:
            current, path = queue.popleft()
            r, c = current

            if len(path) > 30:  # Limit search depth
                continue

            # Check: adjacent to fog cell?
            is_frontier = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = r + move.value[0], c + move.value[1]
                if 0 <= nr < h and 0 <= nc < w:
                    if self.learned_map[nr, nc] == -1:
                        is_frontier = True
                        break

            if path:
                age = step_number - self.visit_map[r, c] if self.learned_map[r, c] == 0 else 0

                if is_frontier or age > 8:
                    # Score: prefer upper-half cells, shorter paths, stale cells
                    upper_bonus = max(0, (h // 2 - r)) * 3  # Bonus for upper rows
                    stale_bonus = age * 2
                    frontier_bonus = 50 if is_frontier else 0
                    distance_penalty = len(path)  # Shorter is better

                    score = upper_bonus + stale_bonus + frontier_bonus - distance_penalty
                    candidates.append((score, path))

            # BFS expand — prioritize UP direction slightly
            for move in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
                nr, nc = r + move.value[0], c + move.value[1]
                npos = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w:
                    if self.learned_map[nr, nc] == 0 and npos not in visited:
                        visited.add(npos)
                        queue.append((npos, path + [move]))

        if candidates:
            # Pick the best scoring candidate
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Absolute fallback
        return self._random_safe_move(my_position)

    def _random_safe_move(self, pos):
        valid = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._max_valid_steps(pos, move, 1) > 0:
                valid.append(move)
        if valid:
            return [random.choice(valid)]
        return [Move.STAY]


class GhostAgent(BaseGhostAgent):
    """Minimal ghost — same as the example."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Test Ghost (Minimal)"
        self.last_known_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos

        if threat is None:
            return self._random_move(my_position, map_state)

        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]
        moves = []
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)

        for move in moves:
            nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] == 0:
                return move

        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] == 0:
                return move
        return Move.STAY

    def _random_move(self, pos, map_state):
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] == 0:
                return move
        return Move.STAY
