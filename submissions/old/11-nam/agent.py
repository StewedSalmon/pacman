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


class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "A Star Pacman"
        self.last_known_enemy_pos = None
        
        # Thêm bộ nhớ để lưu bản đồ tự khám phá
        self.learned_map = None

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # 1. Khởi tạo và cập nhật bản đồ trong não (cực kỳ quan trọng)
        if self.learned_map is None:
            # Bước đầu tiên: copy nguyên cái map hiện tại
            self.learned_map = np.copy(map_state)
        else:
            # Các bước sau: Chỉ lấy những ô nhìn thấy (khác -1) đắp đè lên bản đồ trong não
            visible_mask = map_state != -1
            self.learned_map[visible_mask] = map_state[visible_mask]

        # 2. Cập nhật vị trí Ghost
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        target = enemy_position or self.last_known_enemy_pos
        
        # Nếu mù hoàn toàn, đi random dò đường CHỈ TRÊN NHỮNG Ô HỢP LỆ
        if target is None:
            return self._explore(my_position)

        # 3. Chạy A* TRÊN BẢN ĐỒ ĐÃ GHI NHỚ (self.learned_map)
        path = self.astar(my_position, target, self.learned_map)
        
        if not path or path[0] == Move.STAY:
            return self._explore(my_position)

        # 4. Gom các bước đi liên tiếp
        first_move = path[0]
        desired_steps = 1
        
        for move in path[1:self.pacman_speed]:
            if move == first_move:
                desired_steps += 1
            else:
                break
                
        # Dùng learned_map để check chắc chắn không tông tường
        actual_steps = self._max_valid_steps_memory(my_position, first_move, self.learned_map, desired_steps)
        
        if actual_steps > 0:
            return (first_move, actual_steps)
            
        return (Move.STAY, 1)

    def _explore(self, my_position):
        # Đi random nhưng an toàn dựa trên trí nhớ
        fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        for move in fallback_moves:
            # Thử đi 1 bước xem có an toàn trên bản đồ não không
            if self._max_valid_steps_memory(my_position, move, self.learned_map, 1) > 0:
                return (move, 1)
        return (Move.STAY, 1)

    def _max_valid_steps_memory(self, pos: tuple, move: Move, memory_map: np.ndarray, max_steps: int) -> int:
        steps = 0
        current = pos
        for _ in range(max_steps):
            next_pos = (current[0] + move.value[0], current[1] + move.value[1])
            # Check xem tọa độ có hợp lệ và CÓ TRONG TRÍ NHỚ LÀ Ô TRỐNG KHÔNG
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
                
                # CHỈ ĐƯỢC PHÉP ĐI VÀO CÁC Ô ĐÃ BIẾT LÀ 0
                if 0 <= row < height and 0 <= col < width and memory_map[row, col] == 0:
                    if next_pos not in visited:
                        new_path = path + [move]
                        g_cost = len(new_path)
                        h_cost = heuristic(next_pos)
                        heappush(frontier, (g_cost + h_cost, next_pos, new_path))
                        
        return [Move.STAY]


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "BFS + Monte Carlo Ghost"
        self.learned_map = None
        self.last_known_enemy_pos = None
        self.panic_timer = 0
        self.max_panic_duration = 5  # Thời gian duy trì khoảng cách
        
        # Hyperparameters cho Monte Carlo
        self.mc_simulations = 50
        self.mc_horizon = 12        
        self.assumed_pacman_speed = 1 
        self.last_seen_step = 0

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
        else:
            visible_mask = map_state != -1
            self.learned_map[visible_mask] = map_state[visible_mask]

        if step_number == 1:
            # Kiểm tra xem ô bên phải có phải là đường trống (0) không
            if self._is_valid_move(my_position, Move.RIGHT, self.learned_map):
                self.last_move = Move.RIGHT
                return Move.RIGHT

        if enemy_position is not None:
            # Tính xem Pacman vừa di chuyển bao nhiêu ô trong 1 step
            if self.last_known_enemy_pos is not None and self.last_seen_step == step_number - 1:
                observed_speed = abs(enemy_position[0] - self.last_known_enemy_pos[0]) + abs(enemy_position[1] - self.last_known_enemy_pos[1])
                # Cập nhật tốc độ tối đa từng thấy (nếu thấy nó lướt 2, sẽ nhớ speed là 2)
                if 1 <= observed_speed <= 5:
                    self.assumed_pacman_speed = max(self.assumed_pacman_speed, observed_speed)

            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number
            self.panic_timer = self.max_panic_duration
            current_phase = 2 
            
        elif self.panic_timer > 0:
            self.panic_timer -= 1
            current_phase = 2  
        else:
            current_phase = 1 
            
        if current_phase == 2 and self.last_known_enemy_pos is not None:
            return self._run_survival_tactics(my_position, self.last_known_enemy_pos)
        else:
            return self._run_bfs_explore(my_position)

    
    def _run_bfs_explore(self, my_position: tuple) -> Move:
        queue = deque([(my_position, [])])
        visited = {my_position}
        
        while queue:
            current_pos, path = queue.popleft()
            
            is_frontier = False
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                r = current_pos[0] + move.value[0]
                c = current_pos[1] + move.value[1]
                if 0 <= r < self.learned_map.shape[0] and 0 <= c < self.learned_map.shape[1]:
                    if self.learned_map[r, c] == -1:
                        is_frontier = True
                        break
            
            if is_frontier and len(path) > 0:
                return path[0]
                
            for next_pos, move in self._get_valid_neighbors(current_pos, self.learned_map):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        return self._random_safe_move(my_position)

    
    def _run_survival_tactics(self, my_position: tuple, enemy_pos: tuple) -> Move:
        # Nếu đang đứng chung đường thẳng -> Tìm góc rẽ gấp
        if self._is_in_line_of_sight(my_position, enemy_pos):
            evasion_move = self._bfs_to_break_los(my_position, enemy_pos)
            if evasion_move:
                return evasion_move
                
        # Đã nấp kỹ hoặc không chung đường thẳng -> Dùng MC để trốn
        return self._run_monte_carlo_survival(my_position, enemy_pos)

    def _bfs_to_break_los(self, my_position: tuple, enemy_pos: tuple) -> Move:
        queue = deque([(my_position, [])])
        visited = {my_position}
        
        while queue:
            current_pos, path = queue.popleft()
            
            # Nếu ô này an toàn (khuất tầm nhìn Pacman)
            if current_pos != my_position and not self._is_in_line_of_sight(current_pos, enemy_pos):
                if len(path) > 0:
                    return path[0]
                    
            for next_pos, move in self._get_valid_neighbors(current_pos, self.learned_map):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        return None

    def _run_monte_carlo_survival(self, my_position: tuple, enemy_pos: tuple) -> Move:
        valid_neighbors = self._get_valid_neighbors(my_position, self.learned_map)
        if not valid_neighbors: 
            return Move.STAY
            
        best_move = Move.STAY
        best_score = float('-inf')
        
        for next_pos, move in valid_neighbors:
            # Cấm đi ngược thẳng vào mặt Pacman
            if self._is_in_line_of_sight(next_pos, enemy_pos):
                dist_now = abs(my_position[0] - enemy_pos[0]) + abs(my_position[1] - enemy_pos[1])
                dist_next = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
                if dist_next < dist_now:
                    continue 

            score = 0
            for _ in range(self.mc_simulations):
                score += self._simulate_rollout(next_pos, enemy_pos)
                    
            if score > best_score:
                best_score = score
                best_move = move

            if best_score < 0:
                return self._greedy_evasion(my_position, enemy_pos)
                
        return best_move

    # Mô phỏng các nước đi của Pacman và Ghost ảo để tìm ra phương án tốt nhất
    def _simulate_rollout(self, start_ghost_pos: tuple, start_pacman_pos: tuple) -> float:
        current_g = start_ghost_pos
        current_p = start_pacman_pos
        prev_g = None
        height, width = self.learned_map.shape
        
        for step in range(self.mc_horizon):
            if current_g == current_p: 
                return -1000
                
            # Pacman ảo lướt theo đường thẳng
            best_p_pos = current_p
            best_dist = abs(current_p[0] - current_g[0]) + abs(current_p[1] - current_g[1])
            
            for p_move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                temp_p = current_p
                # Mô phỏng max speed
                for _ in range(self.assumed_pacman_speed): 
                    r = temp_p[0] + p_move.value[0]
                    c = temp_p[1] + p_move.value[1]
                    # Chỉ lướt trên ô đường (0)
                    if 0 <= r < height and 0 <= c < width and self.learned_map[r, c] == 0:
                        temp_p = (r, c)
                        if temp_p == current_g:
                            return -1000 
                    else:
                        break # Đụng tường hoặc unk
                        
                dist = abs(temp_p[0] - current_g[0]) + abs(temp_p[1] - current_g[1])
                if dist < best_dist:
                    best_dist = dist
                    best_p_pos = temp_p
            
            current_p = best_p_pos
            if current_g == current_p: 
                return -1000
                
            # Ghost ảo di chuyển né
            g_neighbors = self._get_valid_neighbors(current_g, self.learned_map)
            g_forward = [n for n in g_neighbors if n[0] != prev_g]
            
            if not g_forward and g_neighbors:
                g_forward = g_neighbors 
                
            if g_forward:
                prev_g = current_g
                current_g = random.choice(g_forward)[0]
                
        # Sống sót được thì thưởng điểm + khoảng cách an toàn
        return 100 + abs(current_g[0] - current_p[0]) + abs(current_g[1] - current_p[1])

    
    # Helper funcs
    # Kiểm tra xem 2 vị trí có chung hàng ngang/dọc và không bị cản bởi tường/unk không
    def _is_in_line_of_sight(self, pos1: tuple, pos2: tuple) -> bool:
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
            return False 
            
        r1, c1 = pos1
        r2, c2 = pos2
        
        if r1 == r2: 
            min_c, max_c = min(c1, c2), max(c1, c2)
            for c in range(min_c + 1, max_c):
                # 1 là tường, -1 là unk -> Cứ khác 0 là chặn tầm nhìn lướt
                if self.learned_map[r1, c] != 0: 
                    return False
            return True
        else: 
            min_r, max_r = min(r1, r2), max(r1, r2)
            for r in range(min_r + 1, max_r):
                if self.learned_map[r, c1] != 0:
                    return False
            return True

    def _get_valid_neighbors(self, pos: tuple, memory_map: np.ndarray) -> list:
        neighbors = []
        height, width = memory_map.shape
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            r = pos[0] + move.value[0]
            c = pos[1] + move.value[1]
            if 0 <= r < height and 0 <= c < width and memory_map[r, c] == 0:
                neighbors.append(((r, c), move))
        return neighbors

    def _random_safe_move(self, pos: tuple) -> Move:
        neighbors = self._get_valid_neighbors(pos, self.learned_map)
        if neighbors:
            return random.choice(neighbors)[1]
        return Move.STAY

    # Chạy ra xa nhất có thể nếu BFS + MC ko cho được nước đi tốt (để giãy là chính)
    def _greedy_evasion(self, my_pos: tuple, enemy_pos: tuple) -> Move:
        best_move = Move.STAY
        best_dist = -1
        for next_pos, move in self._get_valid_neighbors(my_pos, self.learned_map):
            dist = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
            if dist > best_dist:
                best_dist = dist
                best_move = move
        return best_move

    # Kiểm tra xem một nước đi có hợp lệ không 
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)

    # Kiểm tra tọa độ có nằm trong bản đồ và là ô trống (0) không
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    # Tóm lại là có 3 phase chính:
    # 1. Mở map bằng BFS để tìm đường đi mới, tránh bị dồn vào góc chết khi chưa kịp mở đường
    # 2. Khi đã thấy Pacman hoặc vừa mất dấu, chuyển sang kết hợp BFS để tìm đường 
    # thoát khỏi tầm nhìn và Monte Carlo để chọn nước đi sinh tồn tốt nhất dựa trên mô phỏng. 
    # 3. Fallout: Khi cả BFS và MC đều không tìm được nước đi tốt, sẽ dùng greedy (brute force) 
    # để chạy ra xa nhất có thể.
