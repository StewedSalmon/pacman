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
    11 Pacman using speed-aware A*, short-horizon ghost prediction,
    and exploration patrol when no reliable target is available.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "11 Pacman"
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

        # Update persistent world memory.
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
            self.visit_map = np.zeros_like(map_state, dtype=int)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]

        visible = map_state != -1
        self.visit_map[visible] = step_number

        # Track visible ghost and maintain a predicted position.
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

        # Choose pursuit target priority: visible -> predicted -> last known.
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

        # Opening script: RIGHT on turn 1, then UP on turn 2.
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

        # Chase target with speed-aware A*.
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

        # No reliable target: run exploration patrol.
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
    11 Ghost – Counter-meta evasion strategy.

    Exploits the universal Pacman pattern of sweeping top-left first:
    1. Opening book escapes to bottom-right quadrant
    2. Patrols safe junction circuit while Pacman sweeps upper map
    3. Uses phantom Pacman position for always-on awareness
    4. Stochastic scoring prevents deterministic capture patterns
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "11 Ghost"
        self.learned_map = None
        self.last_known_enemy_pos = None
        self.assumed_pacman_speed = 2
        self.last_seen_step = 0
        self.panic_timer = 0
        self.prev_positions = []
        self.last_move_dir = None
        self.capture_distance = 2

        # Keep a light opening/patrol structure for early no-vision phase.
        self._opening_routes = [
            [Move.RIGHT, Move.RIGHT, Move.RIGHT,
             Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN,
             Move.RIGHT, Move.RIGHT],
            [Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT, Move.RIGHT,
             Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN],
        ]
        self._selected_route = None
        self._patrol_nodes = [
            (13, 15), (15, 15), (17, 17),
            (19, 11), (15, 11), (15, 13), (13, 13),
        ]
        self._patrol_idx = 0
        self._patrol_path = []

        self._topology_ready = False
        self._degree = None
        self._junction_dist = {}

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
        if not self._topology_ready:
            self._build_topology()

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

        if self._selected_route is None:
            self._selected_route = random.choice(self._opening_routes)

        # Keep opening route unless we see immediate danger.
        if step_number <= len(self._selected_route):
            book_move = self._selected_route[step_number - 1]
            dist = None
            if enemy_position is not None:
                dist = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])
            if self._is_valid(my_position, book_move) and (dist is None or dist > 4):
                self.last_move_dir = book_move
                return book_move

        if enemy is not None and self.panic_timer > 0:
            candidates = self._pacman_candidates(enemy_position, step_number)
            if not candidates:
                candidates = [enemy]
            move = self._choose_minimax_move(my_position, candidates, step_number)
        else:
            phantom = self._get_phantom_pacman(step_number)
            move = self._patrol_with_awareness(my_position, step_number, phantom)

        self.last_move_dir = move
        return move

    def _build_topology(self):
        h, w = self.learned_map.shape
        self._degree = np.zeros((h, w), dtype=np.int8)
        junctions = []
        for r in range(h):
            for c in range(w):
                if self.learned_map[r, c] == 1:
                    continue
                deg = 0
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nr, nc = r + move.value[0], c + move.value[1]
                    if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] != 1:
                        deg += 1
                self._degree[r, c] = deg
                if deg >= 3:
                    junctions.append((r, c))

        if junctions:
            self._junction_dist = self._multi_source_dist(junctions)
        self._topology_ready = True

    def _multi_source_dist(self, starts):
        dists = {}
        queue = deque(starts)
        for s in starts:
            dists[s] = 0
        while queue:
            cur = queue.popleft()
            d = dists[cur]
            for npos, _ in self._get_neighbors(cur):
                if npos not in dists:
                    dists[npos] = d + 1
                    queue.append(npos)
        return dists

    def _pacman_candidates(self, enemy_position, step_number):
        if enemy_position is not None:
            return [enemy_position]
        if self.last_known_enemy_pos is None:
            return []

        dt = max(1, step_number - self.last_seen_step)
        max_steps = min(8, dt * self.assumed_pacman_speed)
        frontier = {self.last_known_enemy_pos}
        visited = {self.last_known_enemy_pos}
        for _ in range(max_steps):
            nxt = set()
            for pos in frontier:
                nxt.add(pos)
                for npos, _ in self._get_neighbors(pos):
                    if npos not in visited:
                        visited.add(npos)
                        nxt.add(npos)
            frontier = nxt

        cells = list(visited)
        if len(cells) > 18:
            random.shuffle(cells)
            cells = cells[:18]
        return cells

    def _pacman_responses(self, pac_pos):
        responses = [pac_pos]
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            pos = pac_pos
            for _ in range(self.assumed_pacman_speed):
                nr = pos[0] + move.value[0]
                nc = pos[1] + move.value[1]
                if 0 <= nr < self.learned_map.shape[0] and 0 <= nc < self.learned_map.shape[1] and self.learned_map[nr, nc] != 1:
                    pos = (nr, nc)
                    responses.append(pos)
                else:
                    break
        return responses

    def _state_safety(self, ghost_pos, pac_pos, step_number):
        distance = abs(ghost_pos[0] - pac_pos[0]) + abs(ghost_pos[1] - pac_pos[1])
        if distance < self.capture_distance:
            return -1e9

        score = float(distance * 120)
        degree = int(self._degree[ghost_pos[0], ghost_pos[1]]) if self._degree is not None else 2
        if degree <= 1:
            score -= 700
        score += degree * 35

        junction_d = self._junction_dist.get(ghost_pos, 8)
        score -= junction_d * 30

        local_space = self._count_reachable(ghost_pos, 3)
        score += local_space * 10

        visits = sum(1 for p in self.prev_positions[-8:] if p == ghost_pos)
        score -= visits * 25

        if step_number < 30:
            h, w = self.learned_map.shape
            center_dist = abs(ghost_pos[0] - h // 2) + abs(ghost_pos[1] - w // 2)
            score += center_dist * 3

        return score

    def _choose_minimax_move(self, my_pos, pacman_candidates, step_number):
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        if not pacman_candidates:
            pacman_candidates = [self._get_phantom_pacman(step_number)]

        best_move = Move.STAY
        best_worst = float("-inf")
        scored = []

        for next_pos, move in neighbors:
            worst_case = float("inf")
            for pac_guess in pacman_candidates:
                for pac_next in self._pacman_responses(pac_guess):
                    v = self._state_safety(next_pos, pac_next, step_number)
                    if v < worst_case:
                        worst_case = v
                        if worst_case <= -1e8:
                            break
                if worst_case <= -1e8:
                    break
            scored.append((worst_case, move))
            if worst_case > best_worst:
                best_worst = worst_case
                best_move = move

        # Safe stochasticity only among near-optimal moves.
        epsilon = 18.0
        near_best = [m for s, m in scored if s >= best_worst - epsilon and s > -1e8]
        if near_best:
            if self.last_move_dir in near_best and random.random() < 0.55:
                return self.last_move_dir
            return random.choice(near_best)
        return best_move

    def _get_phantom_pacman(self, step_number):
        """Estimate Pacman's likely position based on step count.
        All opposing Pacman agents sweep top-left first from spawn (15,10)."""
        if step_number <= 12:
            return (9, 8)     # Heading up through center
        elif step_number <= 22:
            return (3, 5)     # Sweeping top-left quadrant
        elif step_number <= 32:
            return (3, 15)    # Sweeping top-right quadrant
        elif step_number <= 45:
            return (9, 10)    # Returning through center
        else:
            return (13, 5)    # Sweeping bottom half

    def _patrol_with_awareness(self, my_pos, step_number, phantom_pos):
        """Patrol bottom-right junctions while staying away from phantom Pacman."""
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        h, w = self.learned_map.shape

        # Advance patrol target if we reached it or path is empty.
        current_target = self._patrol_nodes[self._patrol_idx]
        if my_pos == current_target or not self._patrol_path:
            self._patrol_idx = (self._patrol_idx + 1) % len(self._patrol_nodes)
            current_target = self._patrol_nodes[self._patrol_idx]
            self._patrol_path = self._bfs_path(my_pos, current_target)

        # Precompute BFS distances from phantom Pacman.
        phantom_dists = self._bfs_distances(phantom_pos)

        best_move = Move.STAY
        best_score = float('-inf')

        for next_pos, move in neighbors:
            score = 0.0

            # Stay away from phantom Pacman position.
            phantom_d = phantom_dists.get(next_pos, 15)
            score += phantom_d * 30

            # Follow patrol path bonus.
            if self._patrol_path and move == self._patrol_path[0]:
                score += 60

            # Prefer bottom-right quadrant (where Pacman looks last).
            if next_pos[0] >= h // 2:
                score += 15
            if next_pos[1] >= w // 2:
                score += 15

            # Escape route count.
            exits = len(self._get_neighbors(next_pos))
            if exits == 0:
                score -= 10000
            elif exits == 1:
                score -= 500
            score += exits * 10

            # Corridor exposure penalty.
            corridor_len = self._corridor_exposure(next_pos, move)
            score -= corridor_len * 20

            # Avoid revisiting recent cells.
            visits = sum(1 for p in self.prev_positions[-6:] if p == next_pos)
            score -= visits * 30

            # Momentum bonus.
            if self.last_move_dir is not None and move == self.last_move_dir:
                score += 10

            # Stochastic tiebreaker.
            score += random.random() * 5

            if score > best_score:
                best_score = score
                best_move = move

        # Consume patrol path step if we followed it.
        if self._patrol_path and best_move == self._patrol_path[0]:
            self._patrol_path.pop(0)

        return best_move

    def _evade(self, my_pos, enemy_pos, step_number):
        """Choose the next step using weighted survival heuristics with stochasticity."""
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        # Precompute BFS distances from Pacman to all reachable cells.
        pac_dists = self._bfs_distances(enemy_pos)

        best_move = Move.STAY
        best_score = float('-inf')

        # Shuffle to add non-determinism when scores are close.
        neighbor_list = list(neighbors)
        random.shuffle(neighbor_list)

        for next_pos, move in neighbor_list:
            score = 0.0

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
                    score -= 500
                elif los_d <= self.assumed_pacman_speed * 2:
                    score -= 200
                else:
                    score -= 50

            # Bonus for changing direction (less predictable pathing).
            if self.last_move_dir is not None and move != self.last_move_dir:
                score += 20

            # Penalize repeated recent cells to reduce ping-pong loops.
            visits = sum(1 for p in self.prev_positions[-8:] if p == next_pos)
            score -= visits * 25

            # Enhanced corridor exposure penalty.
            corridor_exposure = self._corridor_exposure(next_pos, move)
            score -= corridor_exposure * 25

            # Positional bias depends on game phase.
            h = self.learned_map.shape[0]
            if step_number < 30:
                # Early game: prefer bottom half (away from Pacman's sweep).
                bottom_bonus = max(0, next_pos[0] - h // 2) * 3
                score += bottom_bonus
            else:
                # Late game: just maximize distance, no directional bias.
                pass

            # Stochastic tiebreaker.
            score += random.random() * 5

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
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] != 1:
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

    def _bfs_path(self, start, goal):
        """BFS to find move sequence from start to goal."""
        if start == goal:
            return []
        parent = {start: None}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            for npos, move in self._get_neighbors(cur):
                if npos not in parent:
                    parent[npos] = (cur, move)
                    if npos == goal:
                        # Reconstruct path.
                        path = []
                        node = goal
                        while parent[node] is not None:
                            node, m = parent[node]
                            path.append(m)
                        path.reverse()
                        return path
                    queue.append(npos)
        return []

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
                if self.learned_map[r1, c] == 1:
                    return False
            return True
        else:
            lo, hi = min(r1, r2), max(r1, r2)
            for r in range(lo + 1, hi):
                if self.learned_map[r, c1] == 1:
                    return False
            return True

    def _get_neighbors(self, pos):
        result = []
        h, w = self.learned_map.shape
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] != 1:
                result.append(((nr, nc), move))
        return result

    def _is_valid(self, pos, move):
        nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
        h, w = self.learned_map.shape
        return 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] != 1
