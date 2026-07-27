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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Group 2 Seeker"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_dist = 999
        self.stuck_time = 0
        self.last_known_enemy_pos = None
        self.vision_range = max(1, int(kwargs.get("pacman_obs_radius", 5)))
        self.visit_map = {}

    def _is_valid_position(self, pos, map_state):
        row, col = pos
        h, w = map_state.shape
        # Đảm bảo ô nằm trong biên và không phải là tường (giả định 1 là tường, 0 là đường)
        return 0 <= row < h and 0 <= col < w and map_state[row, col] == 0
    
    def _apply_move(self, pos, move):
        # Apply a move to a position, return new position.
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def step(self, map_state, my_pos, enemy_pos, step_number):
        self.visit_map[my_pos] = self.visit_map.get(my_pos, 0) + 1
        
        can_see_enemy = False
        dist = 999
        if enemy_pos is not None:
            can_see_enemy = True
            self.last_known_enemy_pos = enemy_pos

        # XÁC ĐỊNH MỤC TIÊU VÀ TRẠNG THÁI
        target = None
        is_chasing = False # Biến đánh dấu đang truy đuổi

        if can_see_enemy:
            is_chasing = True
            # Nếu bị vờn (stuck), mục tiêu là ô chặn đầu, nếu không là chính Ghost
            target = self._get_intercept_pos(enemy_pos, my_pos, map_state) if self.stuck_time > 5 else enemy_pos
        elif self.last_known_enemy_pos is not None:
            is_chasing = True # Vẫn coi là đang truy đuổi theo dấu vết
            target = self.last_known_enemy_pos
            if my_pos == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None
                target = self._get_exploration_target(my_pos, map_state)
                is_chasing = False
        else:
            target = self._get_exploration_target(my_pos, map_state)
            is_chasing = False

        # TÌM ĐƯỜNG VỚI TRỌNG SỐ TƯƠNG ỨNG
        path = self._astar(my_pos, target, map_state, is_enemy_target=is_chasing)
        
        # XỬ LÝ KHI A* KHÔNG TÌM THẤY ĐƯỜNG (Để Pacman không đứng yên)
        if not path:
            all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(all_moves)
            for m in all_moves:
                next_p = (my_pos[0] + m.value[0], my_pos[1] + m.value[1])
                if self._is_valid_position(next_p, map_state):
                    return (m, 1)
            return (Move.STAY, 1)

        move = path[0]

        # 4. Tối ưu tốc độ (Speed logic)
        
        if can_see_enemy:
            next_1_step = self._apply_move(my_pos, move)
            next_2_steps = self._apply_move(next_1_step, move)
            if self._is_valid_position(next_1_step, map_state) and self._is_valid_position(next_2_steps, map_state):
                return (move,2)

        return (move, 1)

    def _get_intercept_pos(self, g_pos, p_pos, map_state):
        for dr, dc in [(0, 2), (0, -2), (2, 0), (-2, 0)]:
            target = (g_pos[0] + dr, g_pos[1] + dc)
            if self._is_valid_position(target, map_state):
                return target
        return g_pos

    def _astar(self, start, goal, map_state, is_enemy_target=False):
        # frontier: [f_score, r, c, path]
        frontier = [[abs(start[0]-goal[0]) + abs(start[1]-goal[1]), start[0], start[1], []]]
        visited = {start}
        
        while frontier:
            idx = 0
            for i in range(1, len(frontier)):
                if frontier[i][0] < frontier[idx][0]: idx = i
            
            _, r, c, path = frontier.pop(idx)
            if (r, c) == goal: return path
            
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = r + m.value[0], c + m.value[1]
                if self._is_valid_position((nr, nc), map_state) and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    
                    # TÍNH TOÁN TRỌNG SỐ (G-SCORE)
                    g = len(path) + 1
                    
                    # Nếu KHÔNG PHẢI đuổi Ghost, mới cộng phạt Heatmap để khám phá vùng mới
                    if not is_enemy_target:
                        visit_penalty = self.visit_map.get((nr, nc), 0) * 10
                        g += visit_penalty
                    
                    h = abs(nr-goal[0]) + abs(nc-goal[1])
                    frontier.append([g + h, nr, nc, path + [m]])
        return []

    def _get_exploration_target(self, my_pos, map_state):
        h, w = map_state.shape
        best_target, min_score = None, float('inf')
        for r in range(h):
            for c in range(w):
                if map_state[r, c] == 0:
                    visits = self.visit_map.get((r, c), 0)
                    dist = abs(r - my_pos[0]) + abs(c - my_pos[1])
                    # Trọng số visits cực cao (100) để ép Pacman đi vùng mới
                    score = (visits * 100) + dist
                    if score < min_score:
                        min_score, best_target = score, (r, c)
        return best_target
    
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Group 2 Hider"
        self.depth = 3
        self.vision_range = max(1, int(kwargs.get("ghost_obs_radius", 5)))
        self.last_known_enemy_pos = None
        self.visit_map = {}

    def _get_moves(self, pos, map_state):
        moves = []
        # Ghost thường không nên STAY nếu muốn trốn hiệu quả, 
        # nhưng ta giữ STAY như một lựa chọn cuối cùng
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            np_pos = (pos[0] + m.value[0], pos[1] + m.value[1])
            if 0 <= np_pos[0] < map_state.shape[0] and \
               0 <= np_pos[1] < map_state.shape[1] and \
               map_state[np_pos] == 0:
                moves.append((np_pos, m))
        return moves

    def _minimax(self, g_pos, p_pos, depth, alpha, beta, is_max, map_state):
        """Minimax với Alpha-Beta Pruning để tối ưu tốc độ."""
        if depth == 0 or g_pos == p_pos:
            # Ghost muốn tối đa hóa khoảng cách Manhattan
            return abs(g_pos[0] - p_pos[0]) + abs(g_pos[1] - p_pos[1])

        if is_max: # Ghost lượt (Maximize distance)
            v = -np.inf
            for nxt, _ in self._get_moves(g_pos, map_state):
                v = max(v, self._minimax(nxt, p_pos, depth-1, alpha, beta, False, map_state))
                alpha = max(alpha, v)
                if beta <= alpha: break
            return v
        else: # Pacman lượt (Minimize distance)
            v = np.inf
            for nxt, _ in self._get_moves(p_pos, map_state):
                v = min(v, self._minimax(g_pos, nxt, depth-1, alpha, beta, True, map_state))
                beta = min(beta, v)
                if beta <= alpha: break
            return v

    def _is_dead_end(self, pos, map_state):
        # Không tính STAY là lối thoát
        exits = 0
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] == 0:
                exits += 1
        return exits <= 1

    def step(self, map_state, my_pos, enemy_pos, step_number):
        self.visit_map[my_pos] = self.visit_map.get(my_pos, 0) + 1

        # 1. Cập nhật tầm nhìn và bộ nhớ
        can_see_pacman = False
        if enemy_pos is not None:
            can_see_pacman = True
            self.last_known_enemy_pos = enemy_pos

        # 2. Logic ra quyết định
        # TH1: Thấy Pacman -> Chạy trốn
        if can_see_pacman:
            return self._evade_logic(my_pos, enemy_pos, map_state)
        
        # TH2: Vừa mất dấu -> Chạy tiếp từ vị trí cuối cùng thấy
        if self.last_known_enemy_pos is not None:
            if my_pos == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None
                return self._hide_logic(my_pos, map_state)
            return self._evade_logic(my_pos, self.last_known_enemy_pos, map_state)

        # TH3: Khám phá ẩn nấp bằng Heatmap
        return self._hide_logic(my_pos, map_state)

    def _evade_logic(self, my_pos, p_pos, map_state):
        best_v = -float('inf')
        best_m = Move.STAY
        
        for nxt, m in self._get_moves(my_pos, map_state):
            val = self._minimax(nxt, p_pos, self.depth, -float('inf'), float('inf'), False, map_state)
            
            # Hình phạt ngõ cụt và ô cũ để Ghost linh hoạt
            if self._is_dead_end(nxt, map_state): val -= 10
            val -= self.visit_map.get(nxt, 0) * 0.5 # Trọng số thấp để ko phá vỡ minimax

            # Nếu val bằng nhau, Move.UP -> Move.STAY sẽ được ưu tiên theo thứ tự list
            if val > best_v:
                best_v = val
                best_m = m
        return best_m

    def _hide_logic(self, my_pos, map_state):
        moves = self._get_moves(my_pos, map_state)
        if not moves: return Move.STAY
        
        best_m = Move.STAY
        min_score = float('inf')
        
        for nxt, m in moves:
            if m == Move.STAY: continue
            
            # Score thấp là tốt hơn (ít visit nhất)
            score = self.visit_map.get(nxt, 0) * 10
            if self._is_dead_end(nxt, map_state): score += 50
            
            # Tie-breaking định sẵn: nếu score bằng nhau, giữ nguyên best_m đầu tiên tìm thấy
            if score < min_score:
                min_score = score
                best_m = m
        return best_m