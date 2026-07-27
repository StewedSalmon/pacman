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
- Agents are STATEFUL - you can store memory across steps
- enemy_position may be None when limited observation is enabled
- map_state cells: 1=wall, 0=empty, -1=unseen (fog)
"""

import sys
from pathlib import Path

# Add src to path to import the interface
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
    Interceptor Pacman using speed-aware A*, short-horizon ghost prediction,
    and exploration patrol when no reliable target is available.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Interceptor Pacman"
        self.learned_map = None
        self.visit_map = None
        self.last_known_enemy_pos = None
        self.enemy_history = []          # recent observed ghost positions
        self.predicted_ghost_pos = None
        self.step_count = 0

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):

        self.step_count = step_number

        # 1) Update persistent world memory.
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
            self.visit_map = np.zeros_like(map_state, dtype=int)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]

        visible = map_state != -1
        self.visit_map[visible] = step_number

        # 2) Track visible ghost and maintain a predicted position.
        if enemy_position is not None:
            self.enemy_history.append(enemy_position)
            if len(self.enemy_history) > 8:
                self.enemy_history.pop(0)
            self.last_known_enemy_pos = enemy_position
            self.predicted_ghost_pos = self._predict_ghost(enemy_position)
        else:
            # Ghost not visible: continue prediction from the last known state.
            if self.last_known_enemy_pos is not None:
                self.predicted_ghost_pos = self._predict_ghost(self.last_known_enemy_pos)

        # 3) Choose pursuit target priority: visible -> predicted -> last known.
        target = None
        if enemy_position is not None:
            # Direct chase when ghost is visible.
            target = enemy_position
        elif self.predicted_ghost_pos is not None:
            target = self.predicted_ghost_pos
        elif self.last_known_enemy_pos is not None:
            target = self.last_known_enemy_pos

        # If we reached a stale target and found nothing, clear stale tracking.
        if target is not None and target == my_position and enemy_position is None:
            self.last_known_enemy_pos = None
            self.predicted_ghost_pos = None
            target = None

        # 4) Opening script: RIGHT on turn 1, then UP on turn 2.
        # This quickly positions Pacman toward upper corridors.
        if target is None and step_number <= 2:
            h, w = self.learned_map.shape
            if step_number == 1:
                if self._max_valid_steps(my_position, Move.RIGHT, 1) > 0:
                    return (Move.RIGHT, 1)
            if step_number == 2:
                up_steps = self._max_valid_steps(my_position, Move.UP, self.pacman_speed)
                if up_steps > 0:
                    return (Move.UP, up_steps)

        # 5) Chase target with speed-aware A*.
        if target is not None:
            path = self._speed_aware_astar(my_position, target)
            if path and path[0] != Move.STAY:
                return self._execute_sprint(my_position, path)

            # If direct chase path fails, try cutting off at nearby junctions.
            if enemy_position is not None or (self.last_known_enemy_pos is not None and 
                                                step_number - self._last_seen_step() < 5):
                intercept = self._find_intercept_target(my_position)
                if intercept is not None:
                    path = self._speed_aware_astar(my_position, intercept)
                    if path and path[0] != Move.STAY:
                        return self._execute_sprint(my_position, path)

        # 6) No reliable target: run exploration patrol.
        path = self._smart_patrol(my_position, step_number)
        if path and path[0] != Move.STAY:
            return self._execute_sprint(my_position, path)

        return (Move.STAY, 1)

    # Ghost prediction
    def _predict_ghost(self, last_pos):
        """Project ghost motion from recent observed trajectory."""
        if len(self.enemy_history) < 2:
            return last_pos

        # Estimate average movement vector from recent observations.
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

        # Project two steps forward along the estimated vector.
        pred_r = int(round(last_pos[0] + avg_dr * 2))
        pred_c = int(round(last_pos[1] + avg_dc * 2))

        # Clamp into map bounds.
        h, w = self.learned_map.shape
        pred_r = max(0, min(pred_r, h - 1))
        pred_c = max(0, min(pred_c, w - 1))

        if self.learned_map[pred_r, pred_c] == 0:
            return (pred_r, pred_c)

        # If projected cell is blocked, snap to nearest walkable cell.
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
        """Approximate last seen step using visit timestamps."""
        if self.last_known_enemy_pos is None:
            return 0
        r, c = self.last_known_enemy_pos
        return self.visit_map[r, c]

    def _find_intercept_target(self, my_position):
        """Find a junction target that can intercept likely ghost routes."""
        if not self.enemy_history:
            return None

        ghost_pos = self.enemy_history[-1]

        # Collect nearby junctions (cells with >= 3 exits).
        h, w = self.learned_map.shape
        junctions = []
        search_radius = 6

        for r in range(max(0, ghost_pos[0] - search_radius),
                       min(h, ghost_pos[0] + search_radius + 1)):
            for c in range(max(0, ghost_pos[1] - search_radius),
                           min(w, ghost_pos[1] + search_radius + 1)):
                if self.learned_map[r, c] != 0:
                    continue
                # Count local exits.
                neighbors = 0
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = r + move.value[0], c + move.value[1]
                    if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                        neighbors += 1
                if neighbors >= 3:
                    junctions.append((r, c))

        if not junctions:
            return None

        # Score junctions by whether Pacman can arrive in time.
        ghost_r, ghost_c = ghost_pos
        my_r, my_c = my_position

        best_junction = None
        best_score = float('inf')

        for jr, jc in junctions:
            # Approximate arrival times.
            our_dist = abs(jr - my_r) + abs(jc - my_c)
            ghost_dist = abs(jr - ghost_r) + abs(jc - ghost_c)

            our_turns = our_dist / self.pacman_speed
            ghost_turns = ghost_dist  # ghost speed = 1

            # Keep junctions where Pacman can contest or beat ghost arrival.
            if our_turns <= ghost_turns + 1:
                score = our_dist
                if score < best_score:
                    best_score = score
                    best_junction = (jr, jc)

        return best_junction

    # Speed-aware A*
    def _speed_aware_astar(self, start, goal):
        """A* where a straight sprint up to pacman_speed costs one turn."""

        def heuristic(pos):
            md = abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            # Admissible lower bound in turn units.
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
                # Expand all sprint lengths in a single direction.
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

                    # Any sprint length in one direction consumes one turn.
                    new_g = g_cost + 1
                    if pos not in visited or visited[pos] > new_g:
                        new_path = path + moves_for_this_dir
                        new_f = new_g + heuristic(pos)
                        heappush(frontier, (new_f, new_g, pos, new_path))

        return [Move.STAY]

    # Sprint execution
    def _execute_sprint(self, my_position, path):
        """Compress same-direction prefix into a valid (Move, steps) action."""
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

    # Smart patrol
    def _smart_patrol(self, my_position, step_number):
        """Weighted BFS patrol that favors upper/frontier/stale regions."""

        h, w = self.learned_map.shape

        # Evaluate candidate patrol cells by upper bias, staleness, and fog frontier.

        queue = deque([(my_position, [])])
        visited = {my_position}

        candidates = []  # (score, path)

        while queue:
            current, path = queue.popleft()
            r, c = current

            if len(path) > 30:  # Depth cap for responsiveness.
                continue

            # Frontier test: adjacent to unseen cell.
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
                    # Score combines exploration value and travel cost.
                    upper_bonus = max(0, (h // 2 - r)) * 3
                    stale_bonus = age * 2
                    frontier_bonus = 50 if is_frontier else 0
                    distance_penalty = len(path)

                    score = upper_bonus + stale_bonus + frontier_bonus - distance_penalty
                    candidates.append((score, path))

            # Expansion order gives a mild upward preference.
            for move in [Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN]:
                nr, nc = r + move.value[0], c + move.value[1]
                npos = (nr, nc)
                if 0 <= nr < h and 0 <= nc < w:
                    if self.learned_map[nr, nc] == 0 and npos not in visited:
                        visited.add(npos)
                        queue.append((npos, path + [move]))

        if candidates:
            # Choose highest-scoring patrol target.
            candidates.sort(key=lambda x: x[0], reverse=True)
            return candidates[0][1]

        # Final fallback when no candidate is available.
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
    """
    Marathon Ghost focused on long-term survival.

    Main idea:
    - Maximize BFS path distance from Pacman (real maze distance, not Manhattan).
    - Avoid long straight lanes where a fast Pacman can sprint.
    - Prefer routes with escape options and cover (corners/walls).
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Marathon Ghost"
        self.learned_map = None
        self.last_known_enemy_pos = None
        self.assumed_pacman_speed = 2
        self.last_seen_step = 0
        self.panic_timer = 0
        self.prev_positions = []
        self.last_move_dir = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        # Update persistent map memory from current observation.
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]

        self.prev_positions.append(my_position)
        if len(self.prev_positions) > 20:
            self.prev_positions.pop(0)

        # Track enemy position and estimate sprint speed when visible.
        if enemy_position is not None:
            if (self.last_known_enemy_pos is not None and
                    self.last_seen_step == step_number - 1):
                speed = (abs(enemy_position[0] - self.last_known_enemy_pos[0]) +
                         abs(enemy_position[1] - self.last_known_enemy_pos[1]))
                if 1 <= speed <= 5:
                    self.assumed_pacman_speed = max(self.assumed_pacman_speed, speed)
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number
            self.panic_timer = 15
        elif self.panic_timer > 0:
            self.panic_timer -= 1

        enemy = enemy_position or self.last_known_enemy_pos

        # Opening book move: go right on turn 1 when possible.
        if step_number == 1:
            if self._is_valid(my_position, Move.RIGHT):
                self.last_move_dir = Move.RIGHT
                return Move.RIGHT

        # If threat is active, run evasive scoring; otherwise explore safely.
        if enemy is not None and self.panic_timer > 0:
            move = self._evade(my_position, enemy, step_number)
        else:
            # No recent enemy info: keep exploring while avoiding the center.
            move = self._explore_away_from_center(my_position, step_number)

        self.last_move_dir = move
        return move

    def _evade(self, my_pos, enemy_pos, step_number):
        """Choose the next step using weighted survival heuristics."""
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        # Precompute BFS distances from Pacman to all reachable cells.
        pac_dists = self._bfs_distances(enemy_pos)

        best_move = Move.STAY
        best_score = float('-inf')

        for next_pos, move in neighbors:
            score = 0

            # Path distance from Pacman (highest weight).
            bfs_d = pac_dists.get(next_pos, 0)
            pacman_turns = bfs_d / self.assumed_pacman_speed
            score += pacman_turns * 80

            # Local escape-space bonus.
            reachable = self._count_reachable(next_pos, 4)
            score += reachable * 5

            # Strong dead-end penalties.
            exits = len(self._get_neighbors(next_pos))
            if exits == 0:
                score -= 10000
            elif exits == 1:
                score -= 600

            # Penalize direct line-of-sight exposure.
            if self._is_in_los(next_pos, enemy_pos):
                los_d = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
                if los_d <= self.assumed_pacman_speed:
                    score -= 500  # Can be reached in 1 sprint
                elif los_d <= self.assumed_pacman_speed * 2:
                    score -= 200  # 2 sprints
                else:
                    score -= 50   # Visible but far

            # Small bonus for changing direction (less predictable pathing).
            if self.last_move_dir is not None and move != self.last_move_dir:
                score += 20

            # Penalize repeated recent cells to reduce ping-pong loops.
            visits = sum(1 for p in self.prev_positions[-8:] if p == next_pos)
            score -= visits * 25

            # Penalize long forward corridors (sprint exposure).
            corridor_exposure = self._corridor_exposure(next_pos, move)
            score -= corridor_exposure * 15

            # Slight bias toward upper rows (map-specific prior).
            h = self.learned_map.shape[0]
            upper_bonus = max(0, h // 2 - next_pos[0]) * 2
            score += upper_bonus

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _explore_away_from_center(self, my_pos, step_number):
        """Exploration mode: prefer outer lanes and nearby fog frontiers."""
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        h, w = self.learned_map.shape
        center_r, center_c = h // 2, w // 2

        best_move = Move.STAY
        best_score = float('-inf')

        for next_pos, move in neighbors:
            # Prefer cells farther from map center.
            dist_from_center = abs(next_pos[0] - center_r) + abs(next_pos[1] - center_c)

            # Prefer positions with more short-range mobility.
            reachable = self._count_reachable(next_pos, 3)

            # Bonus if this move touches unexplored fog.
            has_fog = False
            for d in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = next_pos[0] + d.value[0], next_pos[1] + d.value[1]
                if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == -1:
                    has_fog = True
                    break

            # Penalize short back-and-forth loops.
            visits = sum(1 for p in self.prev_positions[-4:] if p == next_pos)

            # Small momentum bonus to keep covering distance.
            momentum = 10 if move == self.last_move_dir else 0

            score = (dist_from_center * 3 + reachable * 2 + 
                     (15 if has_fog else 0) + momentum - visits * 20)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _corridor_exposure(self, pos, direction):
        """Count open cells ahead in one direction (higher means more exposed)."""
        count = 0
        current = pos
        for _ in range(5):
            nr = current[0] + direction.value[0]
            nc = current[1] + direction.value[1]
            h, w = self.learned_map.shape
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                count += 1
                current = (nr, nc)
            else:
                break
        return count

    # BFS utilities

    def _bfs_distances(self, start):
        dists = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            d = dists[cur]
            for npos, _ in self._get_neighbors(cur):
                if npos not in dists:
                    dists[npos] = d + 1
                    queue.append(npos)
        return dists

    def _count_reachable(self, pos, depth):
        visited = {pos}
        frontier = [pos]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for npos, _ in self._get_neighbors(cur):
                    if npos not in visited:
                        visited.add(npos)
                        nxt.append(npos)
            frontier = nxt
        return len(visited) - 1

    def _is_in_los(self, pos1, pos2):
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
            return False
        r1, c1 = pos1
        r2, c2 = pos2
        if r1 == r2:
            lo, hi = min(c1, c2), max(c1, c2)
            for c in range(lo + 1, hi):
                if self.learned_map[r1, c] != 0:
                    return False
            return True
        else:
            lo, hi = min(r1, r2), max(r1, r2)
            for r in range(lo + 1, hi):
                if self.learned_map[r, c1] != 0:
                    return False
            return True

    def _get_neighbors(self, pos):
        result = []
        h, w = self.learned_map.shape
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                result.append(((nr, nc), move))
        return result

    def _is_valid(self, pos, move):
        nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
        h, w = self.learned_map.shape
        return 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0
