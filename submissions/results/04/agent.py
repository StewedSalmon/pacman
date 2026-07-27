import sys
from pathlib import Path
from collections import deque
import heapq
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
class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "TOP 1 Pacman"

        self.memory_map = None
        self.last_enemy_pos = None

        self.history = deque(maxlen=20)

    # ---------- MAIN ----------
    def step(self, map_state, my_pos, enemy_pos, step_number):

        self._update_memory(map_state)
        m = self.memory_map

        if enemy_pos is not None:
            self.last_enemy_pos = enemy_pos

        target = enemy_pos or self.last_enemy_pos

        # ===== LOOP BREAK =====
        if target and self._is_loop(my_pos, target):
            path = self._escape_loop(my_pos, target)
            if path:
                return self._move(path, my_pos)

        # ===== INTERCEPT =====
        if target:
            intercept = self._predict_intercept(my_pos, target)
            if intercept:
                path = self._a_star(my_pos, intercept)
                if path:
                    return self._move(path, my_pos)

        # ===== DIRECT CHASE =====
        if target:
            path = self._a_star(my_pos, target)
            if path:
                return self._move(path, my_pos)

        # ===== EXPLORE =====
        path = self._find_unknown(my_pos)
        if path:
            return self._move(path, my_pos)

        return (Move.STAY, 1)

    # ---------- MOVE ----------
    def _move(self, path, pos):
        move = path[0]

        steps = self._max_steps(pos, move, self.pacman_speed)

        return (move, max(1, steps))

    # ---------- A* ----------
    def _a_star(self, start, goal):
        open_set = []
        counter = 0

        heapq.heappush(open_set, (0, counter, start, []))
        visited = set()

        while open_set:
            _, _, cur, path = heapq.heappop(open_set)

            if cur in visited:
                continue
            visited.add(cur)

            if cur == goal:
                return path

            for move in MOVES:
                nxt = self._next(cur, move)

                if not self._valid(nxt):
                    continue

                new_cost = len(path) + 1
                priority = new_cost + manhattan(nxt, goal)

                counter += 1
                heapq.heappush(open_set, (priority, counter, nxt, path + [move]))

        return None

    # ---------- PREDICT ----------
    def _predict_enemy(self, pos, my_pos):
        best = pos
        best_d = -1

        for move in MOVES:
            nxt = self._next(pos, move)
            if self._valid(nxt):
                d = manhattan(nxt, my_pos)
                if d > best_d:
                    best_d = d
                    best = nxt
        return best

    def _predict_intercept(self, my_pos, enemy_pos):

        g = enemy_pos

        for t in range(1, 7):
            g = self._predict_enemy(g, my_pos)

            path = self._a_star(my_pos, g)
            if path:
                turns = (len(path) + self.pacman_speed - 1) // self.pacman_speed
                if turns <= t:
                    return g
        return None

    # ---------- EXPLORE ----------
    def _find_unknown(self, start):
        q = deque([(start, [])])
        visited = {start}

        while q:
            pos, path = q.popleft()

            if self.memory_map[pos] == -1:
                return path

            for move in MOVES:
                nxt = self._next(pos, move)
                if nxt not in visited and self._valid(nxt):
                    visited.add(nxt)
                    q.append((nxt, path + [move]))

        return None

    # ---------- LOOP ----------
    def _is_loop(self, my_pos, enemy_pos):
        self.history.append((my_pos, enemy_pos))
        return self.history.count((my_pos, enemy_pos)) > 2

    def _escape_loop(self, my_pos, enemy_pos):
        q = deque([(my_pos, [])])
        visited = {my_pos}

        while q:
            pos, path = q.popleft()

            if len(path) > 8:
                continue

            if len(self._neighbors(pos)) >= 3:
                return path

            for move in MOVES:
                nxt = self._next(pos, move)
                if nxt not in visited and self._valid(nxt):
                    visited.add(nxt)
                    q.append((nxt, path + [move]))

        return None

    # ---------- MEMORY ----------
    def _update_memory(self, map_state):
        if self.memory_map is None:
            self.memory_map = np.copy(map_state)
        else:
            mask = map_state != -1
            self.memory_map[mask] = map_state[mask]

    # ---------- UTILS ----------
    def _next(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _valid(self, pos):
        r, c = pos
        h, w = self.memory_map.shape

        if r < 0 or r >= h or c < 0 or c >= w:
            return False

        return self.memory_map[r, c] != 1

    def _neighbors(self, pos):
        return [self._next(pos, m) for m in MOVES if self._valid(self._next(pos, m))]

    def _max_steps(self, pos, move, max_steps):
        steps = 0
        cur = pos

        for _ in range(max_steps):
            nxt = self._next(cur, move)
            if not self._valid(nxt):
                break
            cur = nxt
            steps += 1

        return steps


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
