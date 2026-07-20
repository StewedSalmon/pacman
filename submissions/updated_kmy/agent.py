
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
import time
from collections import deque

def get_mahattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculates the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


# class PacmanAgent(BasePacmanAgent):
#     """
#     Minimax Pacman with BFS evaluation
#     """
#     def __init__(self, **kwargs):
#         """
#         Initialize the Pacman agent.
#         Students can set up any data structures they need here.
#         """
#         super().__init__(**kwargs)
#         self.name = "Minimax Pacman"
#         self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
#         self.max_depth = 3
#         # Memory for limited observation mode
#         self.last_known_enemy_pos = None
#         self.valid_tiles = {}
#         self.map_BFS_cache = {}
        
#     def get_BFS_map(self, start: tuple):
#         """Retrieves or calculates BFS map from start position"""
#         if start in self.map_BFS_cache:
#             return self.map_BFS_cache[start]
        
#         distances = {start: 0}
#         parent = {start: None}
#         queue = deque([start])
#         while queue:
#             current = queue.popleft()
#             for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
#                 x, y = move.value
#                 next = (current[0] + x, current[1] + y)
#                 if self.is_valid_position_precomputed(next) and next not in distances:
#                     distances[next] = distances[current] + 1
#                     parent[next] = (current, move)
#                     queue.append(next)
#         self.map_BFS_cache[start] = (distances, parent)
#         return self.map_BFS_cache[start]
    
#     def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, pacman_turn: bool):
#         # base case
#         if ghost_position == pacman_position:
#            return 0 - depth
#         if depth == 0: 
#             distances, parent = self.get_BFS_map(pacman_position)
#             if ghost_position not in distances:
#                 return float('inf')
            
#             # reconstruct path
#             path = []
#             current = ghost_position
#             while current != pacman_position:
#                 previous, move_taken = parent[current]
#                 path.append(move_taken)
#                 current = previous
#             path.reverse() 
#             if len(path) > 0:
#                 capture_path = path[:-1] # remove the last tile since it is the capture tile
#             else:
#                 capture_path = path
#             # calculate ticks   
#             ticks = self.pacman_tick(capture_path)
#             manhattan_distance = get_mahattan_distance(ghost_position, pacman_position)
#             return (ticks * 10) + len(capture_path) + manhattan_distance * 0.1 # use path length and manhattan distance as tie breakers
        
#         # minimizing for pacman
#         if pacman_turn:
#             min_eval = float('inf')
#             pacman_next_positions = self.get_sprint_options(pacman_position)         
#             for _, _, next_pacman_pos in pacman_next_positions:
#                 eval = self.minimax(ghost_position, next_pacman_pos, map_state, depth - 1, alpha, beta, False)
#                 min_eval = min(min_eval, eval)
#                 beta = min(beta, eval)
#                 if beta <= alpha:
#                     break
#             return min_eval
            
#         # maximizing for ghost
#         else:
#             max_eval = -float('inf')
#             for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
#                 next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                
#                 if self.is_valid_position_precomputed(next_ghost_pos):
#                     eval = self.minimax(next_ghost_pos, pacman_position, map_state, depth - 1, alpha, beta, True)
#                     max_eval = max(max_eval, eval)
#                     alpha = max(alpha, eval)
#                     if beta <= alpha:
#                         break
#             return max_eval
        
#     def precompute_valid_tiles(self, map_state: np.ndarray):
#         """Precompute valid tiles for quick access during minimax."""
#         self.valid_tiles = {}
#         height, width = map_state.shape
#         for row in range(height):
#             for col in range(width):
#                 if map_state[row, col] == 0:
#                     self.valid_tiles[(row, col)] = True
    
#     def is_valid_position_precomputed(self, pos: tuple):
#         """Check if a position is valid using precomputed tiles."""
#         return pos in self.valid_tiles
    
#     def pacman_tick(self, path: list) -> int:
#         if not path:
#             return 0
#         ticks = 0
#         current_move = path[0]
#         steps = 0
        
#         for move in path:
#             if move == current_move and steps < self.pacman_speed:
#                 steps += 1
#             else:
#                 ticks += 1
#                 current_move = move
#                 steps = 1
#         if steps > 0:
#             ticks += 1
#         return ticks
    
#     def get_sprint_options(self, pos: tuple):
#         """
#         Generates all valid moves and step counts for Pacman.
#         Returns a list of tuples: (Move, step_count, landing_position)
#         """
#         options = []
#         for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
#             delta_row, delta_col = move.value
#             current_pos = pos        
#             for steps in range(1, self.pacman_speed + 1):
#                 next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
#                 # Stop projecting if we hit a wall
#                 if not self.is_valid_position_precomputed(next_pos):
#                     break 
#                 current_pos = next_pos
#                 options.append((move, steps, current_pos))
                
#         return options
    
#     def step(self, map_state: np.ndarray, 
#              my_position: tuple, 
#              enemy_position: tuple,
#              step_number: int):
#         # Update memory if enemy is visible
#         if enemy_position is not None:
#             self.last_known_enemy_pos = enemy_position
        
#         # Use current sighting, fallback to last known, or explore
#         target = enemy_position or self.last_known_enemy_pos
        
#         if target is None:
#             # No information about enemy - explore randomly
#             return self._explore(my_position)
        
#         if len(self.valid_tiles) == 0:
#             self.precompute_valid_tiles(map_state)
        
#         best_move = Move.STAY
#         best_steps = 1
#         min_eval = float('inf')
#         alpha = -float('inf')
#         beta = float('inf')
        
#         possible_moves = self.get_sprint_options(my_position)
#         for move, steps, landing_pos in possible_moves:
#             eval = self.minimax(enemy_position, landing_pos, map_state, self.max_depth, alpha, beta, False)
#             if eval < min_eval:
#                 min_eval = eval
#                 best_move = move
#                 best_steps = steps
#             beta = min(beta, eval)
            
#         if best_steps > 1:
#             return (best_move, best_steps)
#         return (best_move, 1)

#     def _explore(self, my_position: tuple):
#         """Random exploration when enemy position is unknown."""
#         all_moves = self.get_sprint_options(my_position)
#         if all_moves:
#             move, steps, _ = random.choice(all_moves)
#             if steps > 1:
#                 return (move, steps)
#             return (move, 1)
#         return (Move.STAY, 1)
    
#     def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
#         """Check if a position is valid (not a wall and within bounds)."""
#         row, col = pos
#         height, width = map_state.shape
        
#         if row < 0 or row >= height or col < 0 or col >= width:
#             return False
        
#         return map_state[row, col] == 0

class TimeoutException(Exception):
    """Exception tùy chỉnh dùng để ngắt thuật toán khi sắp hết thời gian."""
    pass

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Speed Demon Ghost V2"
        self.pacman_speed = 2
        self.valid_tiles = set()
        self.dist_cache = {} 
        self.dead_ends = set() # [THÊM MỚI] Bộ nhớ nhận diện ngõ cụt
        self.time_budget = 0.85

    def precompute_valid_tiles(self, map_state: np.ndarray):
        self.valid_tiles = set()
        self.dist_cache.clear()
        self.dead_ends.clear()
        
        height, width = map_state.shape
        for row in range(height):
            for col in range(width):
                if map_state[row, col] == 0:
                    self.valid_tiles.add((row, col))
                    
        # [THÊM MỚI] Xây dựng đồ thị để quét toàn bộ ngõ cụt trên bản đồ
        graph = {pos: [] for pos in self.valid_tiles}
        for pos in self.valid_tiles:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (pos[0] + move.value[0], pos[1] + move.value[1])
                if nxt in self.valid_tiles:
                    graph[pos].append(nxt)
                    
        # Loại bỏ dần các lá (nhánh chỉ có 1 lối đi) để tìm ra các hành lang cụt
        leaves = deque([pos for pos, neighbors in graph.items() if len(neighbors) <= 1])
        while leaves:
            curr = leaves.popleft()
            self.dead_ends.add(curr)
            for neighbor in graph[curr]:
                if neighbor not in self.dead_ends:
                    exits = sum(1 for n in graph[neighbor] if n not in self.dead_ends)
                    if exits <= 1:
                        leaves.append(neighbor)

    def get_bfs_dist(self, source: tuple, target: tuple) -> int:
        if source not in self.dist_cache:
            dist = {source: 0}
            queue = deque([source])
            while queue:
                curr = queue.popleft()
                d = dist[curr]
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    nxt = (curr[0] + move.value[0], curr[1] + move.value[1])
                    if nxt in self.valid_tiles and nxt not in dist:
                        dist[nxt] = d + 1
                        queue.append(nxt)
            self.dist_cache[source] = dist
            
        return self.dist_cache[source].get(target, 999)

    def get_mobility(self, pos: tuple) -> int:
        return sum(1 for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                   if (pos[0] + move.value[0], pos[1] + move.value[1]) in self.valid_tiles)

    def minimax(self, ghost_pos: tuple, pacman_pos: tuple, depth: int, alpha: float, beta: float, ghost_turn: bool, deadline: float):
        if time.time() > deadline:
            raise TimeoutException()

        if abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) < 2:
            return -100000 - depth  
            
        if depth == 0:
            dist_to_pac = self.get_bfs_dist(ghost_pos, pacman_pos)
            mobility = self.get_mobility(ghost_pos)
            
            score = (dist_to_pac * 10.0) + (mobility * 3.0)
            
            # [THÊM MỚI] Trừ điểm cực nặng nếu Ghost chui vào ngõ cụt
            if ghost_pos in self.dead_ends:
                score -= 5000.0
                
            return score

        if ghost_turn:
            max_eval = -float('inf')
            
            # [THÊM MỚI] Move Ordering: Ưu tiên duyệt các nước đi xa Pacman nhất trước
            moves = []
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                nxt = ghost_pos if m == Move.STAY else (ghost_pos[0] + m.value[0], ghost_pos[1] + m.value[1])
                if nxt in self.valid_tiles or m == Move.STAY:
                    # Dùng khoảng cách Manhattan thô để sắp xếp nhanh
                    dist_to_threat = abs(nxt[0] - pacman_pos[0]) + abs(nxt[1] - pacman_pos[1])
                    moves.append((dist_to_threat, nxt, m))
            
            moves.sort(key=lambda x: x[0], reverse=True) # Xa nhất lên đầu
            
            for _, nxt, m in moves:
                eval_score = self.minimax(nxt, pacman_pos, depth - 1, alpha, beta, False, deadline)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
            
        else:
            min_eval = float('inf')
            pac_moves = [pacman_pos]
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                curr = pacman_pos
                for step in range(self.pacman_speed):
                    nxt = (curr[0] + move.value[0], curr[1] + move.value[1])
                    if nxt not in self.valid_tiles:
                        break
                    curr = nxt
                    pac_moves.append(curr)
                    
            # [THÊM MỚI] Move Ordering: Giả lập Pacman sẽ ưu tiên các ô gần Ghost nhất trước
            pac_moves = list(set(pac_moves)) 
            pac_moves.sort(key=lambda p: abs(p[0] - ghost_pos[0]) + abs(p[1] - ghost_pos[1]))
            
            for next_pac in pac_moves:
                eval_score = self.minimax(ghost_pos, next_pac, depth - 1, alpha, beta, True, deadline)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is None:
            return Move.STAY

        if not self.valid_tiles:
            self.precompute_valid_tiles(map_state)

        deadline = time.time() + self.time_budget
        best_move_overall = Move.STAY
        
        possible_moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            nxt = my_position if move == Move.STAY else (my_position[0] + move.value[0], my_position[1] + move.value[1])
            if nxt in self.valid_tiles or move == Move.STAY:
                possible_moves.append((move, nxt))

        possible_moves.sort(key=lambda x: self.get_bfs_dist(x[1], enemy_position), reverse=True)

        for depth in range(1, 20):
            try:
                best_move_current_depth = possible_moves[0][0]
                max_eval = -float('inf')
                alpha = -float('inf')
                beta = float('inf')

                for move, nxt in possible_moves:
                    eval_score = self.minimax(nxt, enemy_position, depth, alpha, beta, False, deadline)
                    if eval_score > max_eval:
                        max_eval = eval_score
                        best_move_current_depth = move
                    alpha = max(alpha, eval_score)

                best_move_overall = best_move_current_depth
                
                if max_eval < -50000:
                    break

            except TimeoutException:
                break

        return best_move_overall