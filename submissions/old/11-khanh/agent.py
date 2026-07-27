"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
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


class PacmanAgent(BasePacmanAgent):
    """
    Bloodhound Pacman (Seeker) Agent - Nâng cấp khả năng đánh hơi và tuần tra
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Bloodhound Pacman"
        self.last_known_enemy_pos = None
        self.learned_map = None
        # Thêm trí nhớ về thời gian để "đánh hơi" vùng cũ
        self.visit_map = None 

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # 1. Khởi tạo và cập nhật bản đồ + thời gian nhìn thấy
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
            self.visit_map = np.zeros_like(map_state, dtype=int)
        else:
            visible_mask = map_state != -1
            self.learned_map[visible_mask] = map_state[visible_mask]
            
        # Ghi nhận bước (step) hiện tại cho tất cả các ô đang nằm trong tầm nhìn
        visible_mask = map_state != -1
        self.visit_map[visible_mask] = step_number

        # 2. Xử lý mục tiêu (Ghost)
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            target = enemy_position
        else:
            target = self.last_known_enemy_pos

        # Đã đến nơi mà không thấy Ghost -> Xóa dấu vết để chuyển sang chế độ đánh hơi
        if target == my_position and enemy_position is None:
            self.last_known_enemy_pos = None
            target = None

        # 3. Quyết định hướng đi
        if target is None:
            # Dùng BFS đánh hơi vùng tối HOẶC vùng đã lâu không nhìn tới
            path = self._smart_patrol(my_position, step_number)
        else:
            # Rượt đuổi chớp nhoáng bằng A*
            path = self.astar(my_position, target, self.learned_map)
            if not path or path[0] == Move.STAY:
                path = self._smart_patrol(my_position, step_number)

        # 4. Thực thi Sprint (Lướt nhiều ô trong 1 step)
        if path and path[0] != Move.STAY:
            first_move = path[0]
            desired_steps = 1
            
            for move in path[1:self.pacman_speed]:
                if move == first_move:
                    desired_steps += 1
                else:
                    break
                    
            actual_steps = self._max_valid_steps_memory(my_position, first_move, self.learned_map, desired_steps)
            
            if actual_steps > 0:
                return (first_move, actual_steps)

        return (Move.STAY, 1)

    def _smart_patrol(self, my_position: tuple, step_number: int):
        """BFS quét vùng biên hoặc các ô 'cũ' để tìm Ghost ẩn nấp"""
        queue = deque([(my_position, [])])
        visited = set([my_position])
        
        while queue:
            current_pos, path = queue.popleft()
            r, c = current_pos
            
            # Kiểm tra 1: Có kề ô sương mù (-1) không?
            is_frontier = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr = r + move.value[0]
                nc = c + move.value[1]
                if 0 <= nr < self.learned_map.shape[0] and 0 <= nc < self.learned_map.shape[1]:
                    if self.learned_map[nr, nc] == -1:
                        is_frontier = True
                        break
                        
            # Kiểm tra 2: Ô này có phải là ô trống mà đã qua 15 bước chưa ai nhìn vào không?
            is_stale = (self.learned_map[r, c] == 0) and (step_number - self.visit_map[r, c] > 15)
            
            # Nếu thỏa mãn 1 trong 2 -> Đi tới đó
            if (is_frontier or is_stale) and len(path) > 0:
                return path 
                
            # Mở rộng BFS tìm kiếm tiếp
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr = r + move.value[0]
                nc = c + move.value[1]
                next_pos = (nr, nc)
                
                if 0 <= nr < self.learned_map.shape[0] and 0 <= nc < self.learned_map.shape[1]:
                    # Chỉ đi dạo trên các ô đã biết là đường (0)
                    if self.learned_map[nr, nc] == 0 and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, path + [move]))
                        
        # Fallback an toàn nếu bản đồ quá nhỏ hoặc vừa nhìn hết mọi ngóc ngách
        return self._random_safe_move(my_position)

    def _random_safe_move(self, pos: tuple):
        valid_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._max_valid_steps_memory(pos, move, self.learned_map, 1) > 0:
                valid_moves.append(move)
        if valid_moves:
            return [random.choice(valid_moves)]
        return [Move.STAY]

    def _max_valid_steps_memory(self, pos: tuple, move: Move, memory_map: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            next_pos = (current[0] + move.value[0], current[1] + move.value[1])
            row, col = next_pos
            height, width = memory_map.shape
            if row < 0 or row >= height or col < 0 or col >= width or memory_map[row, col] != 0:
                break
            steps += 1
            current = next_pos
        return steps

    def astar(self, start, goal, memory_map):
        from heapq import heappush, heappop
        
        def heuristic(pos):
            return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])
            
        frontier = [(0, start, [])]
        visited = set()
        
        while frontier:
            f_cost, current_pos, path = heappop(frontier)
            
            if current_pos == goal:
                return path
                
            if current_pos in visited:
                continue
                
            visited.add(current_pos)
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = (current_pos[0] + move.value[0], current_pos[1] + move.value[1])
                row, col = next_pos
                height, width = memory_map.shape
                
                if 0 <= row < height and 0 <= col < width and memory_map[row, col] == 0:
                    if next_pos not in visited:
                        new_path = path + [move]
                        g_cost = len(new_path)
                        h_cost = heuristic(next_pos)
                        heappush(frontier, (g_cost + h_cost, next_pos, new_path))
                        
        return [Move.STAY]


class GhostAgent(BaseGhostAgent):
    """
    Example Ghost agent using a simple evasive strategy.
    Students should implement their own search algorithms here.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Example Evasive Ghost"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Simple evasive strategy: move away from Pacman.
        
        When enemy_position is None (limited observation mode),
        uses last known position or moves randomly.
        
        Students should implement better search algorithms like:
        - BFS to find furthest point
        - A* to plan escape route
        - Minimax for adversarial search
        - etc.
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            return self._random_move(my_position, map_state)
        
        # Calculate direction away from threat
        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]
        
        # List of possible moves in order of preference
        moves = []
        
        # Prioritize vertical movement away from Pacman
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement away from Pacman
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
        for move in moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            # Check if move is valid
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no preferred move is valid, try any valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no move is valid, stay
        return Move.STAY

    def _random_move(self, my_position: tuple, map_state: np.ndarray) -> Move:
        """Random movement when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            if self._is_valid_position(new_pos, map_state):
                return move
        
        return Move.STAY
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
