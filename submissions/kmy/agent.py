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


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "A* Speed Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
        
        # 3. Dùng thuật toán A* tìm đường ngắn nhất đến mục tiêu
        path = self.a_star_search(my_position, target, map_state)
        
        # Nếu không tìm được đường (hoặc Ghost bị bao vây/tọa độ lỗi), chuyển sang thám hiểm
        if not path or path[0] == Move.STAY:
            if my_position == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None
            return self._explore(my_position, map_state)
            
        # 4. Trích xuất nước đi và tối ưu hóa tốc độ thẳng (Sprint)
        first_move = path[0]
        
        intended_steps = 0
        for move in path:
            if move == first_move and intended_steps < self.pacman_speed:
                intended_steps += 1
            else:
                break
                
        # Validate số bước thực tế để tránh lao vào tường
        valid_steps = self._max_valid_steps(my_position, first_move, map_state, intended_steps)
        
        if valid_steps > 0:
            return (first_move, valid_steps)
            
        return (Move.STAY, 1)
    
    # Helper methods (you can add more)
    
    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Thuật toán Breadth-First Search tìm đường đi ngắn nhất.
        Trả về danh sách các Move để đi từ start đến goal.
        """
        from collections import deque
        
        # Hàng đợi chứa tuple: (vị_trí_hiện_tại, con_đường_đã_đi_để_đến_đây)
        queue = deque([(start, [])])
        visited = set()
        visited.add(start)
        
        while queue:
            current_pos, path = queue.popleft()
            
            # Nếu đã đến đích, trả về con đường
            if current_pos == goal:
                return path
                
            # Duyệt qua 4 hướng
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                
                # Nếu vị trí tiếp theo hợp lệ và chưa từng đi qua
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        return [] # Trả về rỗng nếu không tìm thấy đường
    
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        """Hàm Heuristic: Tính khoảng cách Manhattan giữa 2 điểm."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def a_star_search(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """
        Thuật toán A* Search.
        Trả về danh sách các Move để đi từ start đến goal một cách tối ưu.
        """
        import heapq
        import itertools
        
        # Bộ đếm để phá vỡ thế hòa (tie-breaker) khi 2 node có cùng điểm f(n)
        # Giúp heapq không bao giờ cố gắng so sánh trực tiếp các tuple hay object Move
        tie_breaker = itertools.count()
        
        # Priority Queue lưu trữ các node theo định dạng: 
        # (điểm_f, ID_tie_breaker, điểm_g, vị_trí_hiện_tại, con_đường_đã_đi)
        open_set = []
        heapq.heappush(open_set, (0, next(tie_breaker), 0, start, []))
        
        # Dictionary lưu chi phí g(n) tốt nhất để đến được 1 ô, giúp tránh đi lòng vòng
        best_g_scores = {start: 0}
        
        while open_set:
            # Lấy ra ô có tổng chi phí dự kiến f(n) thấp nhất
            f, _, g, current_pos, path = heapq.heappop(open_set)
            
            # Đã đến đích -> trả về đường đi
            if current_pos == goal:
                return path
                
            # Mở rộng các hướng đi
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                delta_row, delta_col = move.value
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                
                # Nếu vị trí tiếp theo không phải là tường
                if self._is_valid_position(next_pos, map_state):
                    new_g = g + 1 # Chi phí thực tế đi thêm 1 bước
                    
                    # Chỉ đưa vào hàng đợi nếu đây là lần đầu tiên ta đến ô này, 
                    # hoặc con đường mới này ngắn hơn con đường trước đó ta từng tìm ra
                    if next_pos not in best_g_scores or new_g < best_g_scores[next_pos]:
                        best_g_scores[next_pos] = new_g
                        h = self._manhattan_distance(next_pos, goal)
                        new_f = new_g + h # f(n) = g(n) + h(n)
                        
                        heapq.heappush(open_set, (new_f, next(tie_breaker), new_g, next_pos, path + [move]))
                        
        return [] # Trả về rỗng nếu hoàn toàn không có đường đến đích


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Alpha-Beta Ghost"
        self.search_depth = 4
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        # self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):

        best_move = Move.STAY
        best_score = float('-inf')
        alpha = float('-inf')
        beta = float('inf')
        
        legal_moves = self._get_legal_moves(my_position, map_state)
        # Hết đường thì đứng chờ
        if not legal_moves:
            return Move.STAY
            
        for move in legal_moves:
            next_pos = self._apply_move(my_position, move)
            
            # Tuyệt đối không tự sát
            if next_pos == enemy_position:
                continue
                
            score = self.alpha_beta(next_pos, enemy_position, self.search_depth - 1, alpha, beta, False, map_state)
            
            if score > best_score:
                best_score = score
                best_move = move
                
            alpha = max(alpha, best_score)
            
        # Nếu đường nào cũng chết, chọn đại 1 đường để di chuyển thay vì chết tại chỗ
        if best_move is None or best_score == float('-inf'):
            import random
            return random.choice(legal_moves) if legal_moves else Move.STAY
            
        return best_move
    
    # Helper methods (you can add more)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def alpha_beta(self, ghost_pos, pacman_pos, depth, alpha, beta, is_maximizing, map_state):
        if depth == 0 or ghost_pos == pacman_pos:
            return self.evaluate(ghost_pos, pacman_pos, map_state)
            
        if is_maximizing: 
            max_eval = float('-inf')
            for move in self._get_legal_moves(ghost_pos, map_state):
                next_pos = self._apply_move(ghost_pos, move)
                eval = self.alpha_beta(next_pos, pacman_pos, depth - 1, alpha, beta, False, map_state)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break 
            return max_eval
            
        else: 
            min_eval = float('inf')
            for move in self._get_legal_moves(pacman_pos, map_state):
                next_pos = self._apply_move(pacman_pos, move)
                eval = self.alpha_beta(ghost_pos, next_pos, depth - 1, alpha, beta, True, map_state)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break 
            return min_eval

    def evaluate(self, ghost_pos, pacman_pos, map_state):
        if ghost_pos == pacman_pos:
            return -99999
            
        dist = abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1])
        legal_moves = self._get_legal_moves(ghost_pos, map_state)
        mobility = len(legal_moves)
        
        score = dist * 10
        if mobility <= 1:
            score -= 100 
        elif mobility >= 3:
            score += 20  
            
        return score

    def _get_legal_moves(self, pos: tuple, map_state: np.ndarray) -> list:
        moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                moves.append(move)
        return moves

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
