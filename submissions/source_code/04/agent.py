import sys
from pathlib import Path
from collections import deque
from heapq import heappop, heappush
import numpy as np
import random

# Add src
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


# ===================== UTILS =====================
MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# ===================== PACMAN (TOP 1) =====================
DIRECTIONS = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
PATROL_DIRECTIONS = (Move.UP, Move.LEFT, Move.RIGHT, Move.DOWN)
LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.### # ###.#####",
    "    #.#       #.#    ",
    "#####.# ##-## #.#####",
    "     .  . G .  .     ",
    "#####.# ##### #.#####",
    "    #.#       #.#    ",
    "#####.# ##### #.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....P.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]


class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))

        self.learned_map, _, ghost_start = self._build_map()
        self.visit_map = None
        self.step_count = 0

        self.last_known_enemy_pos = ghost_start
        self.predicted_enemy_pos = ghost_start
        self.enemy_history = [ghost_start] if ghost_start else []
        self.state_history = deque(maxlen=16)
        self.distance_cache = {}

    def step(
        self,
        map_state: np.ndarray,
        my_position: tuple,
        enemy_position: tuple,
        step_number: int,
    ):
        self.step_count = step_number
        self._sync_state(map_state, enemy_position, step_number)

        target = self._select_target(my_position, enemy_position)
        if target is not None:
            chase_action = self._plan_chase(my_position, target, enemy_position)
            if chase_action is not None:
                return chase_action

        patrol_path = self._patrol(my_position, step_number)
        return self._path_to_action(my_position, patrol_path)

    def _sync_state(self, map_state, enemy_position, step_number):
        if self.visit_map is None:
            self.visit_map = np.zeros_like(map_state, dtype=int)

        self.distance_cache.clear()
        visible_mask = map_state != -1
        self.learned_map[visible_mask] = map_state[visible_mask]
        self.visit_map[visible_mask] = step_number

        if enemy_position is None:
            if self.last_known_enemy_pos is not None:
                self.predicted_enemy_pos = self._predict_enemy(self.last_known_enemy_pos)
            return

        self.enemy_history.append(enemy_position)
        if len(self.enemy_history) > 8:
            self.enemy_history.pop(0)

        self.last_known_enemy_pos = enemy_position
        self.predicted_enemy_pos = self._predict_enemy(enemy_position)

    def _select_target(self, my_position, enemy_position):
        target = enemy_position or self.predicted_enemy_pos or self.last_known_enemy_pos
        if target == my_position and enemy_position is None:
            self.last_known_enemy_pos = None
            self.predicted_enemy_pos = None
            return None
        return target

    def _plan_chase(self, my_position, target, enemy_position):
        if enemy_position is not None:
            self.state_history.append((my_position, enemy_position))
            direct_path = self._speed_aware_astar(my_position, enemy_position)
            direct_action = self._path_to_action(my_position, direct_path) if direct_path else None
            cutoff_target = self._find_cutoff_target(my_position, enemy_position)
            if cutoff_target is not None:
                cutoff_path = self._speed_aware_astar(my_position, cutoff_target)
                if cutoff_path is not None:
                    cutoff_action = self._path_to_action(my_position, cutoff_path)
                    if not self._should_skip_cutoff(
                        my_position,
                        enemy_position,
                        cutoff_action,
                        direct_action,
                    ):
                        return cutoff_action

            planned_action = self._adversarial_action(my_position, enemy_position)
            if planned_action is not None:
                return planned_action

        path = self._speed_aware_astar(my_position, target)
        if path:
            return self._path_to_action(my_position, path)

        recent_contact = (
            self.last_known_enemy_pos is not None
            and self.step_count - self._last_seen_step() < 5
        )
        if enemy_position is None and not recent_contact:
            return None

        intercept_target = self._find_intercept_target(my_position)
        if intercept_target is None:
            return None

        intercept_path = self._speed_aware_astar(my_position, intercept_target)
        if not intercept_path:
            return None
        return self._path_to_action(my_position, intercept_path)

    def _predict_enemy(self, last_pos):
        if len(self.enemy_history) < 2:
            return last_pos

        row_delta = 0
        col_delta = 0
        for previous, current in zip(self.enemy_history, self.enemy_history[1:]):
            row_delta += current[0] - previous[0]
            col_delta += current[1] - previous[1]

        samples = len(self.enemy_history) - 1
        predicted = (
            int(round(last_pos[0] + 2 * row_delta / samples)),
            int(round(last_pos[1] + 2 * col_delta / samples)),
        )
        return self._nearest_walkable(predicted, fallback=last_pos)

    def _last_seen_step(self):
        if self.last_known_enemy_pos is None or self.visit_map is None:
            return 0
        row, col = self.last_known_enemy_pos
        return self.visit_map[row, col]

    def _find_intercept_target(self, my_position):
        if not self.enemy_history:
            return None

        ghost_row, ghost_col = self.enemy_history[-1]
        best_target = None
        best_score = -float("inf")

        for junction in self._nearby_junctions((ghost_row, ghost_col), radius=6):
            our_distance = self._path_distance(my_position, junction)
            ghost_distance = self._path_distance((ghost_row, ghost_col), junction)

            if our_distance == float("inf") or ghost_distance == float("inf"):
                continue
            if self._turns_to_reach(our_distance) > ghost_distance + 1:
                continue
            escape_routes = len(self._walkable_neighbors(junction))
            score = ghost_distance * 8 - our_distance * 5 - escape_routes * 20
            if score > best_score:
                best_score = score
                best_target = junction

        return best_target

    def _nearby_junctions(self, center, radius):
        height, width = self.learned_map.shape
        row_start = max(0, center[0] - radius)
        row_end = min(height, center[0] + radius + 1)
        col_start = max(0, center[1] - radius)
        col_end = min(width, center[1] + radius + 1)

        junctions = []
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                position = (row, col)
                if not self._is_walkable(position):
                    continue
                exits = sum(1 for move in DIRECTIONS if self._can_step(position, move))
                if exits >= 3:
                    junctions.append(position)
        return junctions

    def _speed_aware_astar(self, start, goal):
        if start == goal:
            return []

        frontier = []
        tie_breaker = 0
        heappush(frontier, (self._turn_heuristic(start, goal), 0, tie_breaker, start, []))
        best_cost = {}

        while frontier:
            _, turns, _, position, path = heappop(frontier)
            if position == goal:
                return path
            if position in best_cost and best_cost[position] <= turns:
                continue

            best_cost[position] = turns
            for move in DIRECTIONS:
                sprint_path = []
                current = position
                for _ in range(self.pacman_speed):
                    next_position = self._next_position(current, move)
                    if not self._is_walkable(next_position):
                        break

                    current = next_position
                    sprint_path.append(move)
                    next_turns = turns + 1
                    if best_cost.get(current, float("inf")) <= next_turns:
                        continue

                    tie_breaker += 1
                    heappush(
                        frontier,
                        (
                            next_turns + self._turn_heuristic(current, goal),
                            next_turns,
                            tie_breaker,
                            current,
                            path + sprint_path,
                        ),
                    )

        return None

    def _turn_heuristic(self, start, goal):
        distance = self._path_distance(start, goal)
        if distance == float("inf"):
            return float("inf")
        return self._turns_to_reach(distance)

    def _path_to_action(self, my_position, path):
        if not path:
            return (Move.STAY, 1)

        move = path[0]
        steps = 1
        for next_move in path[1 : self.pacman_speed]:
            if next_move != move:
                break
            steps += 1

        valid_steps = self._max_valid_steps(my_position, move, steps)
        return (move, valid_steps) if valid_steps > 0 else (Move.STAY, 1)

    def _max_valid_steps(self, start, move, limit):
        steps = 0
        position = start
        for _ in range(limit):
            next_position = self._next_position(position, move)
            if not self._is_walkable(next_position):
                break
            position = next_position
            steps += 1
        return steps

    def _patrol(self, my_position, step_number):
        queue = deque([(my_position, [])])
        visited = {my_position}
        candidates = []

        while queue:
            position, path = queue.popleft()
            if len(path) > 30:
                continue

            if path:
                score = self._patrol_score(position, path, step_number)
                if score is not None:
                    candidates.append((score, path))

            for move in PATROL_DIRECTIONS:
                next_position = self._next_position(position, move)
                if next_position in visited or not self._is_walkable(next_position):
                    continue
                visited.add(next_position)
                queue.append((next_position, path + [move]))

        if candidates:
            return max(candidates, key=lambda item: item[0])[1]

        valid_moves = [move for move in DIRECTIONS if self._can_step(my_position, move)]
        return [choice(valid_moves)] if valid_moves else []

    def _patrol_score(self, position, path, step_number):
        row, col = position
        age = step_number - self.visit_map[row, col]
        is_frontier = any(
            self._in_bounds(self._next_position(position, move))
            and self.learned_map[self._next_position(position, move)] == -1
            for move in DIRECTIONS
        )

        if not is_frontier and age <= 8:
            return None

        height = self.learned_map.shape[0]
        upper_bias = max(0, height // 2 - row) * 3
        stale_bias = age * 2
        frontier_bias = 50 if is_frontier else 0
        distance_penalty = len(path)
        return upper_bias + stale_bias + frontier_bias - distance_penalty

    def _nearest_walkable(self, position, fallback):
        height, width = self.learned_map.shape
        row = min(max(position[0], 0), height - 1)
        col = min(max(position[1], 0), width - 1)
        clamped = (row, col)

        if self._is_walkable(clamped):
            return clamped

        best_position = fallback
        best_distance = float("inf")
        for row_offset in range(-2, 3):
            for col_offset in range(-2, 3):
                candidate = (row + row_offset, col + col_offset)
                if not self._is_walkable(candidate):
                    continue
                distance = self._manhattan(candidate, clamped)
                if distance < best_distance:
                    best_distance = distance
                    best_position = candidate
        return best_position

    def _find_cutoff_target(self, pacman_pos, ghost_pos):
        if not self._is_stuck_chasing():
            return None

        best_target = None
        best_score = -float("inf")
        escape_route = self._predict_escape_route(ghost_pos, pacman_pos, horizon=16)

        for turns_ahead, candidate in enumerate(escape_route[1:], start=1):
            our_distance = self._path_distance(pacman_pos, candidate)
            if our_distance == float("inf"):
                continue

            our_turns = self._turns_to_reach(our_distance)
            if our_turns > turns_ahead:
                continue

            exits = len(self._walkable_neighbors(candidate))
            local_area = self._reachable_area(candidate, limit=6)
            wait_bonus = (turns_ahead - our_turns) * 30
            trap_bonus = (4 - exits) * 45 + max(0, 18 - local_area) * 4
            score = wait_bonus + trap_bonus - our_distance * 4

            if score > best_score:
                best_score = score
                best_target = candidate

        return best_target

    def _adversarial_action(self, pacman_pos, ghost_pos):
        pacman_actions = self._legal_pacman_actions(pacman_pos)
        if not pacman_actions:
            return None

        best_action = None
        best_score = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        search_depth = 4

        for action in pacman_actions:
            next_pacman = self._apply_pacman_action(pacman_pos, action)
            score = self._ghost_turn(next_pacman, ghost_pos, search_depth - 1, alpha, beta)
            if score > best_score:
                best_score = score
                best_action = action
            alpha = max(alpha, best_score)

        if best_action is None:
            return None
        return best_action

    def _predict_escape_route(self, ghost_pos, pacman_pos, horizon):
        route = [ghost_pos]
        current = ghost_pos
        seen = {ghost_pos: 1}

        for _ in range(horizon):
            next_position = max(
                self._walkable_neighbors(current) or [current],
                key=lambda candidate: (
                    self._path_distance(pacman_pos, candidate),
                    self._reachable_area(candidate, limit=6),
                    -seen.get(candidate, 0),
                ),
            )
            route.append(next_position)
            seen[next_position] = seen.get(next_position, 0) + 1
            current = next_position

        return route

    def _ghost_turn(self, pacman_pos, ghost_pos, depth, alpha, beta):
        if self._is_capture_state(pacman_pos, ghost_pos):
            return 10_000 + depth
        if depth == 0:
            return self._evaluate_state(pacman_pos, ghost_pos)

        worst_score = float("inf")
        for move in self._legal_ghost_moves(ghost_pos):
            next_ghost = self._next_position(ghost_pos, move)
            score = self._pacman_turn(pacman_pos, next_ghost, depth - 1, alpha, beta)
            worst_score = min(worst_score, score)
            beta = min(beta, worst_score)
            if beta <= alpha:
                break
        return worst_score

    def _pacman_turn(self, pacman_pos, ghost_pos, depth, alpha, beta):
        if self._is_capture_state(pacman_pos, ghost_pos):
            return 10_000 + depth
        if depth == 0:
            return self._evaluate_state(pacman_pos, ghost_pos)

        best_score = -float("inf")
        for action in self._legal_pacman_actions(pacman_pos):
            next_pacman = self._apply_pacman_action(pacman_pos, action)
            score = self._ghost_turn(next_pacman, ghost_pos, depth - 1, alpha, beta)
            best_score = max(best_score, score)
            alpha = max(alpha, best_score)
            if beta <= alpha:
                break
        return best_score

    def _evaluate_state(self, pacman_pos, ghost_pos):
        if self._is_capture_state(pacman_pos, ghost_pos):
            return 10_000

        path_distance = self._path_distance(pacman_pos, ghost_pos)
        if path_distance == float("inf"):
            return -10_000

        ghost_moves = len(self._legal_ghost_moves(ghost_pos))
        pacman_moves = len(self._legal_pacman_actions(pacman_pos))
        ghost_space = self._reachable_area(ghost_pos, limit=10)
        pacman_space = self._reachable_area(pacman_pos, limit=10)
        repeated_states = self.state_history.count((pacman_pos, ghost_pos))

        return (
            -120 * path_distance
            - 35 * ghost_moves
            - 4 * ghost_space
            + 8 * pacman_moves
            + 2 * pacman_space
            - 180 * repeated_states
        )

    def _is_stuck_chasing(self):
        if len(self.state_history) < 6:
            return False

        recent_states = list(self.state_history)[-6:]
        distances = [self._path_distance(pacman, ghost) for pacman, ghost in recent_states]
        same_gap = len(set(distances)) == 1 and distances[0] <= 3
        aligned = all(
            pacman[0] == ghost[0] or pacman[1] == ghost[1]
            for pacman, ghost in recent_states
        )
        repeated_states = len(set(recent_states)) <= 4
        pacman_positions = [pacman for pacman, _ in recent_states]
        ghost_positions = [ghost for _, ghost in recent_states]
        return same_gap and aligned and (
            repeated_states
            or self._is_back_and_forth(pacman_positions)
            or self._is_back_and_forth(ghost_positions)
        )

    def _legal_pacman_actions(self, position):
        actions = []
        for move in DIRECTIONS:
            max_steps = self._max_valid_steps(position, move, self.pacman_speed)
            for steps in range(1, max_steps + 1):
                actions.append((move, steps))
        if not actions:
            actions.append((Move.STAY, 1))
        return actions

    def _legal_ghost_moves(self, position):
        moves = [move for move in DIRECTIONS if self._can_step(position, move)]
        return moves or [Move.STAY]

    def _apply_pacman_action(self, position, action):
        move, steps = action
        current = position
        for _ in range(steps):
            next_position = self._next_position(current, move)
            if not self._is_walkable(next_position):
                break
            current = next_position
        return current

    def _is_capture_state(self, pacman_pos, ghost_pos):
        return self._manhattan(pacman_pos, ghost_pos) < 1

    def _path_distance(self, start, goal):
        key = (start, goal)
        if key in self.distance_cache:
            return self.distance_cache[key]

        if start == goal:
            self.distance_cache[key] = 0
            self.distance_cache[(goal, start)] = 0
            return 0

        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            position, distance = queue.popleft()
            for neighbor in self._walkable_neighbors(position):
                if neighbor in visited:
                    continue
                if neighbor == goal:
                    self.distance_cache[key] = distance + 1
                    self.distance_cache[(goal, start)] = distance + 1
                    return distance + 1
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))

        self.distance_cache[key] = float("inf")
        self.distance_cache[(goal, start)] = float("inf")
        return float("inf")

    def _reachable_area(self, start, limit):
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            position, depth = queue.popleft()
            if depth == limit:
                continue
            for neighbor in self._walkable_neighbors(position):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append((neighbor, depth + 1))
        return len(visited)

    def _walkable_neighbors(self, position):
        return [
            self._next_position(position, move)
            for move in DIRECTIONS
            if self._can_step(position, move)
        ]

    def _turns_to_reach(self, path_distance):
        return (path_distance + self.pacman_speed - 1) // self.pacman_speed

    def _should_skip_cutoff(self, pacman_pos, ghost_pos, cutoff_action, direct_action):
        if cutoff_action is None or direct_action is None:
            return False

        cutoff_move, _ = cutoff_action
        direct_move, _ = direct_action
        if not self._is_reverse_move(cutoff_move, direct_move):
            return False

        current_distance = self._path_distance(pacman_pos, ghost_pos)
        direct_distance = self._path_distance(
            self._apply_pacman_action(pacman_pos, direct_action),
            ghost_pos,
        )
        cutoff_distance = self._path_distance(
            self._apply_pacman_action(pacman_pos, cutoff_action),
            ghost_pos,
        )
        return direct_distance < current_distance and cutoff_distance >= direct_distance

    def _is_back_and_forth(self, positions):
        if len(positions) < 4:
            return False
        return (
            positions[-1] == positions[-3]
            and positions[-2] == positions[-4]
            and positions[-1] != positions[-2]
        )

    def _is_reverse_move(self, first_move, second_move):
        return (
            first_move.value[0] == -second_move.value[0]
            and first_move.value[1] == -second_move.value[1]
        )

    def _build_map(self):
        height = len(LAYOUT)
        width = len(LAYOUT[0])
        grid = np.zeros((height, width), dtype=int)
        pacman_start = None
        ghost_start = None

        for row, line in enumerate(LAYOUT):
            for col, cell in enumerate(line):
                if cell in {"#", "-"}:
                    grid[row, col] = 1
                    continue

                if cell == "P":
                    pacman_start = (row, col)
                elif cell == "G":
                    ghost_start = (row, col)

        return grid, pacman_start, ghost_start

    def _can_step(self, position, move):
        return self._is_walkable(self._next_position(position, move))

    def _next_position(self, position, move):
        return position[0] + move.value[0], position[1] + move.value[1]

    def _is_walkable(self, position):
        if not self._in_bounds(position):
            return False
        return self.learned_map[position] == 0

    def _in_bounds(self, position):
        row, col = position
        height, width = self.learned_map.shape
        return 0 <= row < height and 0 <= col < width

    @staticmethod
    def _manhattan(first, second):
        return abs(first[0] - second[0]) + abs(first[1] - second[1])



# ===================== GHOST =====================
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent with hybrid evasion algorithms.
        """
        super().__init__(**kwargs)
        self.name = "Hybrid Ghost"
        
        self.minimax_depth = 3
        self.mc_simulations = 15
        self.mc_depth = 10
        self.memory_map = np.full((21, 21), -1, dtype=int)

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        visible_mask = map_state != -1
        self.memory_map[visible_mask] = map_state[visible_mask]
        valid_moves = self._get_valid_moves(my_position, self.memory_map)
        
        if not valid_moves:
            return Move.STAY
        if enemy_position is None:
            return self._explore_unknown(my_position, valid_moves)
        
        dist_to_enemy = self._manhattan_distance(my_position, enemy_position)
        
        # CHIẾN THUẬT HYBRID: Tùy theo khoảng cách để chọn thuật toán
        if dist_to_enemy <= 4:
            # Gần kẻ thù: Cần tính toán chính xác tuyệt đối từng bước
            best_move = self._minimax_move(self.memory_map, my_position, enemy_position, valid_moves)
        else:
            # Kẻ thù ở xa: Dùng Monte Carlo để tìm các khu vực rộng lớn dài hạn
            best_move = self._monte_carlo_move(self.memory_map, my_position, enemy_position, valid_moves)

        return best_move
    
    # 1. THUẬT TOÁN MINIMAX
    def _minimax_move(self, map_state, my_pos, enemy_pos, valid_moves):
        best_move = random.choice(valid_moves)
        best_val = -float('inf')
        alpha = -float('inf')
        beta = float('inf')

        for move in valid_moves:
            next_pos = self._apply_move(my_pos, move)
            # Sau khi Ghost đi, đến lượt Enemy (is_maximizing = False)
            move_val = self._minimax(map_state, next_pos, enemy_pos, self.minimax_depth - 1, False, alpha, beta)
            
            if move_val > best_val:
                best_val = move_val
                best_move = move
            alpha = max(alpha, best_val)
            
        return best_move

    def _minimax(self, map_state, ghost_pos, enemy_pos, depth, is_maximizing, alpha, beta):
        dist = self._manhattan_distance(ghost_pos, enemy_pos)
        if depth == 0 or dist == 0:
            freedom = len(self._get_valid_moves(ghost_pos, map_state))
            return dist + freedom * 0.5

        if is_maximizing:
            max_eval = -float('inf')
            for move in self._get_valid_moves(ghost_pos, map_state):
                next_pos = self._apply_move(ghost_pos, move)
                eval = self._minimax(map_state, next_pos, enemy_pos, depth - 1, False, alpha, beta)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            enemy_moves = self._get_valid_moves(enemy_pos, map_state)
            if not enemy_moves:
                return dist + len(self._get_valid_moves(ghost_pos, map_state)) * 0.5
            enemy_moves.sort(key=lambda m: self._manhattan_distance(self._apply_move(enemy_pos, m), ghost_pos))
            
            for move in enemy_moves:
                next_pos = self._apply_move(enemy_pos, move)
                eval = self._minimax(map_state, ghost_pos, next_pos, depth - 1, True, alpha, beta)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval

    # 2. THUẬT TOÁN MONTE CARLO
    def _monte_carlo_move(self, map_state, my_pos, enemy_pos, valid_moves):
        best_move = random.choice(valid_moves)
        best_avg_score = -float('inf')

        for move in valid_moves:
            total_score = 0
            for _ in range(self.mc_simulations):
                total_score += self._simulate_random_playout(map_state, self._apply_move(my_pos, move), enemy_pos)
            
            avg_score = total_score / self.mc_simulations
            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_move = move
                
        return best_move

    def _simulate_random_playout(self, map_state, initial_ghost_pos, initial_enemy_pos):
        g_pos = initial_ghost_pos
        e_pos = initial_enemy_pos
        
        for _ in range(self.mc_depth):
            if g_pos == e_pos:
                return -100
            g_moves = self._get_valid_moves(g_pos, map_state)
            if not g_moves: break
            g_pos = self._apply_move(g_pos, random.choice(g_moves))
            
            # Kẻ thù đi greedy (tiến về phía Ghost)
            e_moves = self._get_valid_moves(e_pos, map_state)
            if not e_moves: break
            best_e_move = min(e_moves, key=lambda m: self._manhattan_distance(self._apply_move(e_pos, m), g_pos))
            e_pos = self._apply_move(e_pos, best_e_move)
            
        return self._manhattan_distance(g_pos, e_pos)
    
    def _explore_unknown(self, start_pos: tuple, valid_moves: list) -> Move:
        """Dùng BFS để tìm đường đi ngắn nhất hướng tới khu vực chưa khám phá (-1)."""
        queue = deque([(start_pos, [])])
        visited = {start_pos}
        height, width = self.memory_map.shape

        while queue:
            curr_pos, path = queue.popleft()
            r, c = curr_pos
            
            # Kiểm tra xem ô hiện tại có nằm cạnh ô chưa biết (-1) nào không
            is_near_unknown = False
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if self.memory_map[nr, nc] == -1:
                        is_near_unknown = True
                        break
            
            # Nếu tìm thấy mép của vùng chưa biết, trả về bước đi ĐẦU TIÊN để hướng tới đó
            if is_near_unknown and len(path) > 0:
                return path[0]
                
            # Mở rộng BFS tìm kiếm các ô trống (0)
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = self._apply_move(curr_pos, move)
                if self._is_valid_position(next_pos, self.memory_map) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        # Fallback: Nếu đã mở 100% map mà vẫn không thấy địch, chuyển sang đi ngẫu nhiên
        return random.choice(valid_moves)
    
    # CÁC HÀM TIỆN ÍCH
    def _get_valid_moves(self, pos: tuple, map_state: np.ndarray) -> list:
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            new_pos = self._apply_move(pos, move)
            if self._is_valid_position(new_pos, map_state):
                valid_moves.append(move)
        return valid_moves

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] == 0

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
