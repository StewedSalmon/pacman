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
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random

import pacmanAlgorithm

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

        self.name = "Fog Walker Pacman"

        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.internal_map = np.full((21, 21), -1, dtype=int)  # -1 = unseen, 0 = empty, 1 = wall

        self.last_ghost_dir = None       # Lưu quán tính (hướng đi) cuối cùng của Ghost
        self.predicted_target = None     # Tọa độ dự đoán Ghost sẽ tới

        # Dồn góc
        self.locked_target = None
        self.lock_counter = 0

        # Chạy khi hết sương mù
        self.patrol_target = None

        self.heatmap = np.zeros((21, 21), dtype=int)

        # Quyết định hướng càn quét ngay từ khi game bắt đầu
        self.rush_direction = "RIGHT" if random.random() <= 0.36 else "LEFT"
    
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
        
        try:
            # Cập nhật bản đồ 
            self._update_memory(map_state)
            
            # Cập nhật vị trí, hướng đi của Ghost
            if enemy_position is not None:
                if self.last_known_enemy_pos is not None and self.last_known_enemy_pos != enemy_position:
                    # Ghi nhớ hướng đi cuối cùng của Ghost
                    self.last_ghost_dir = (enemy_position[0] - self.last_known_enemy_pos[0], 
                                           enemy_position[1] - self.last_known_enemy_pos[1])
                self.last_known_enemy_pos = enemy_position
                self.predicted_target = None # Hủy dự đoán cũ khi nhìn thấy trực tiếp

            # Xóa dấu vết khi đã tới nơi 
            if self.last_known_enemy_pos is not None:
                if my_position == self.last_known_enemy_pos or my_position == self.predicted_target:
                    self.last_known_enemy_pos = None
                    self.last_ghost_dir = None
                    self.predicted_target = None

            # Các trạng thái hành vi chính:
            target_pos = None
            current_state = 0
            explore_path = []

            if enemy_position is not None:
                # Đuổi khi thấy trực tiếp 
                current_state = 1
                target_pos = enemy_position
                
                # Bật Dồn góc sớm từ step 15
                if step_number > 15 and self.last_ghost_dir is not None:
                    if self.lock_counter > 0 and self.locked_target is not None:
                        target_pos = self.locked_target
                        self.lock_counter -= 1
                        if my_position == self.locked_target:
                            self.lock_counter = 0
                            self.locked_target = None
                            target_pos = enemy_position
                    else:
                        choke_point = self._get_forward_choke_point(enemy_position, self.last_ghost_dir, self.internal_map)
                        target_pos = choke_point
                        if choke_point != enemy_position:
                            self.locked_target = choke_point
                            self.lock_counter = 3
                            
            elif self.last_known_enemy_pos is not None:
                # Đuổi theo dấu vết cũ 
                current_state = 2
                
                if self.predicted_target is None:
                    self.predicted_target = self._predict_ghost_target(
                        self.last_known_enemy_pos, 
                        self.last_ghost_dir, 
                        self.internal_map, 
                        predict_steps=3
                    )
                
                target_pos = self.predicted_target
                self.lock_counter = 0
                self.locked_target = None
                
            else:
                # TRẠNG THÁI 3: KHÁM PHÁ / TUẦN TRA
                current_state = 3
                
                target_pos, explore_path = self._find_best_frontier(my_position)
                
                if target_pos is None:
                    # Cần tìm điểm mới
                    if (self.patrol_target is None or 
                        my_position == self.patrol_target or 
                        self.heatmap[self.patrol_target[0], self.patrol_target[1]] == 0):
                        
                        valid_cells_mask = (self.internal_map == 0)
                        
                        if np.any(valid_cells_mask):
                            # --- CẬP NHẬT: THUẬT TOÁN ĐA DẠNG HÓA KHÔNG GIAN (SPATIAL DIVERSITY) ---
                            
                            # 1. Lấy tất cả tọa độ hợp lệ và nhiệt độ tương ứng
                            valid_coords = np.argwhere(valid_cells_mask)
                            heats = self.heatmap[valid_coords[:, 0], valid_coords[:, 1]]
                            
                            # 2. Sắp xếp giảm dần theo nhiệt độ
                            sorted_indices = np.argsort(heats)[::-1]
                            sorted_coords = valid_coords[sorted_indices]
                            
                            candidates = []
                            min_dist = 8  # Khoảng cách bắt buộc để được coi là "cách xa nhau"
                            
                            # 3. Lọc ra 3 "Đỉnh núi" nhiệt độ nằm rải rác trên map
                            for coord in sorted_coords:
                                is_far_enough = True
                                for cand in candidates:
                                    # Tính khoảng cách Manhattan
                                    dist = abs(coord[0] - cand[0]) + abs(coord[1] - cand[1])
                                    if dist < min_dist:
                                        is_far_enough = False
                                        break
                                
                                if is_far_enough:
                                    candidates.append((coord[0], coord[1]))
                                
                                # Lấy đủ 3 đỉnh núi thì dừng
                                if len(candidates) >= 3:
                                    break
                            
                            # Fallback: Nếu map quá nhỏ không tìm đủ 3 điểm xa nhau, lấy tạm điểm nóng nhất
                            if not candidates:
                                candidates.append((sorted_coords[0][0], sorted_coords[0][1]))
                                
                            # 4. Chọn ngẫu nhiên 1 trong 3 đỉnh núi này
                            self.patrol_target = random.choice(candidates)
                    
                    target_pos = self.patrol_target
                    _, explore_path = self._bfs_to_target(my_position, target_pos)

                self.lock_counter = 0
                self.locked_target = None

            # Tìm đường đi đến target_pos
            path = []
            if target_pos is not None:
                if current_state == 3:
                    # Đang khám phá/tuần tra -> Dùng luôn đường đi của BFS
                    path = explore_path
                else:
                    # Đang rượt Ghost -> Dùng A*
                    path = pacmanAlgorithm.astar(my_position, target_pos, self.internal_map)
                    
                    if not path and current_state == 1 and target_pos != enemy_position:
                        target_pos = enemy_position
                        self.lock_counter = 0
                        self.locked_target = None
                        path = pacmanAlgorithm.astar(my_position, enemy_position, self.internal_map)

                    # BFS khi A* bị kẹt hoặc để tuần tra
                    if not path and target_pos != my_position:
                        _, path = self._bfs_to_target(my_position, target_pos)

            # Hành động theo đường đi tìm được
            if path:
                best_move = path[0]
                consecutive_steps = 1
                for i in range(1, len(path)):
                    if path[i] == best_move:
                        consecutive_steps += 1
                    else:
                        break
                
                desired_steps = min(consecutive_steps, self.pacman_speed)
                steps = self._max_valid_steps(my_position, best_move, self.internal_map, desired_steps)
                
                if steps > 0:
                    return (best_move, steps)
                    
            # Fallback cuối cùng
            fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            random.shuffle(fallback_moves) # --- FIX PING-PONG: Lắc xí ngầu hướng đi ---
            for move in fallback_moves:
                if self._is_valid_move(my_position, move, self.internal_map):
                    return (move, 1)
            
            return (Move.STAY, 1)

        except Exception as e:
            print(f"LỖI PACMAN: {e}")
            return (Move.STAY, 1)
        
    # Helper methods
    
    def _update_memory(self, map_state: np.ndarray):
        """Cập nhật bộ nhớ và Bản đồ nhiệt (Heatmap)"""
        # 1. Cập nhật bản đồ sương mù
        self.internal_map = np.where(map_state != -1, map_state, self.internal_map)
        
        # 2. Tăng nhiệt độ (độ cũ) của toàn bộ bản đồ lên 1 (Nơi nào càng lâu không nhìn thấy càng nóng)
        self.heatmap += 1
        
        # 3. Làm nguội (reset về 0) những ô đang nằm trong tầm nhìn (khác -1)
        self.heatmap = np.where(map_state != -1, 0, self.heatmap)

    def _predict_ghost_target(self, start_pos, ghost_dir, map_state, predict_steps=3):
        """
        Giả lập đường chạy của Ghost.
        Phóng 1 điểm ảo di chuyển theo ghost_dir. Nếu đụng tường thì tự động bẻ lái.
        """
        if ghost_dir is None:
            return start_pos
            
        curr_pos = start_pos
        curr_dir = ghost_dir
        
        for _ in range(predict_steps):
            next_pos = (curr_pos[0] + curr_dir[0], curr_pos[1] + curr_dir[1])
            
            # Nếu phía trước là đường trống -> đi tiếp
            if self._is_valid_position(next_pos, map_state):
                curr_pos = next_pos
            else:
                # Nếu đụng tường -> Tìm ngã rẽ (tuyệt đối không quay đầu)
                opposite_dir = (-curr_dir[0], -curr_dir[1])
                valid_turns = []
                
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    if move.value != opposite_dir:
                        test_pos = (curr_pos[0] + move.value[0], curr_pos[1] + move.value[1])
                        if self._is_valid_position(test_pos, map_state):
                            valid_turns.append((test_pos, move.value))
                
                if valid_turns:
                    # Giả định Ghost rẽ vào ngã rẽ hợp lệ đầu tiên
                    curr_pos, curr_dir = valid_turns[0]
                else:
                    # Ngõ cụt, Ghost phải dừng lại
                    break
                    
        return curr_pos

    def _get_forward_choke_point(self, ghost_pos, ghost_dir, map_state):
        """Phóng tia Raycast tìm ngã tư chặn đầu Ghost"""
        curr_pos = ghost_pos
        for _ in range(7): 
            next_pos = (curr_pos[0] + ghost_dir[0], curr_pos[1] + ghost_dir[1])
            if not self._is_valid_position(next_pos, map_state):
                return curr_pos 
            
            valid_neighbors = 0
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                n = (next_pos[0] + move.value[0], next_pos[1] + move.value[1])
                if self._is_valid_position(n, map_state):
                    valid_neighbors += 1
            if valid_neighbors > 2:
                return next_pos # Tìm thấy ngã tư
            curr_pos = next_pos
        return curr_pos

    def _find_best_frontier(self, my_pos):
        """
        HEURISTIC FRONTIER SEARCH: Chiến thuật càn quét cho Map cố định.
        Gom tất cả vùng biên và chấm điểm để ép Pacman 'Rush' lên nửa trên bản đồ.
        """
        queue = deque([(my_pos, [])])
        visited = {my_pos}
        height, width = self.internal_map.shape
        
        frontiers = [] # Danh sách chứa TẤT CẢ các vùng sương mù tìm được
        
        # 1. QUÉT BFS TOÀN BỘ VÙNG SÁNG (Rất nhanh vì chỉ có vài chục/trăm ô 0)
        while queue:
            curr_pos, path = queue.popleft()
            
            is_frontier = False
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
                if 0 <= nr < height and 0 <= nc < width:
                    if self.internal_map[nr, nc] == -1:
                        is_frontier = True
                        break
                        
            if is_frontier:
                frontiers.append((curr_pos, path))
            else:
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                    dr, dc = move.value
                    next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                    if self._is_valid_position(next_pos, self.internal_map) and next_pos not in visited:
                        visited.add(next_pos)
                        queue.append((next_pos, path + [move]))
                        
        if not frontiers:
            return None, []
            
        # 2. CHẤM ĐIỂM (HEURISTIC SCORING) ĐỂ TÌM VÙNG BIÊN NGON NHẤT
        best_frontier_pos = None
        best_frontier_path = []
        best_score = -float('inf')
        
        for f_pos, f_path in frontiers:
            score = 0
            
            # Tiêu chí 1: Khoảng cách (Trừ 1.5 điểm cho mỗi bước đi)
            # Giữ cho Pacman không chạy đi chạy lại giữa 2 đầu bản đồ
            score -= len(f_path) * 1.5 
            
            # Tiêu chí 2: Trọng lực phía Bắc (Cộng 3 điểm cho mỗi hàng nhích lên trên)
            # Row 0 là đỉnh map. Càng gần đỉnh, điểm cộng càng khổng lồ.   
            # Ghost thường spawn ở phía trên, đây là đòn 'Rush Top' chí mạng.
            score += (21 - f_pos[0]) * 3
            
            #Tiêu chí 3.1: Kiểm soát Trung tuyến (Cộng 8 điểm)
            #Nếu sương mù nằm ở khu vực giữa map (cột 6 đến 14) -> Ưu tiên quét để chiếm tầm nhìn
            # if 6 <= f_pos[1] <= 14:
            #     score += 8

            # #Tiêu chí 3.3: CHIẾN THUẬT LỆCH PHẢI (RIGHT-BIAS) 
            # #1. Tạo độ nghiêng nhẹ sang Phải
            # if f_pos[1] > 10:
            #     score += 4    # Nửa bên Phải: Cộng 4 điểm
            # elif f_pos[1] < 10:
            #     score -= 1    # Nửa bên Trái: Trừ nhẹ 1 điểm để tạo độ dốc
                
            # # 2. Hút về khu vực trung tâm (Cột 6 đến 14) để giữ tầm nhìn bao quát
            # if 6 <= f_pos[1] <= 14:
            #     score += 3

            # Tiêu chí 3.4: CHIẾN THUẬT LỆCH Trái 
            # 1. Tạo độ nghiêng nhẹ sang Trái
            # if f_pos[1] < 10:
            #     score += 4    # Nửa bên Trái: Cộng 4 điểm
            # elif f_pos[1] > 10:
            #     score -= 1    # Nửa bên Phải: Trừ nhẹ 1 điểm để tạo độ dốc
                
            # # 2. Hút về khu vực trung tâm (Cột 6 đến 14) để giữ tầm nhìn bao quát
            # if 6 <= f_pos[1] <= 14:
            #     score += 3

            # Tiêu chí 3.4: CHIẾN THUẬT 30 PHẢI / 70 TRÁI
            if self.rush_direction == "RIGHT":
                # Kịch bản 30% số trận: Hút mạnh về cánh Phải
                if f_pos[1] >= 11:
                    score += 5
                elif f_pos[1] <= 9:
                    score -= 2
            else:
                # Kịch bản 70% số trận: Hút mạnh về cánh Trái
                if f_pos[1] <= 9:
                    score += 5
                elif f_pos[1] >= 11:
                    score -= 2 
            # Tiêu chí bổ trợ: Vẫn giữ lực hút về khu vực trung tâm để tối ưu tia nhìn 5 ô
            if 6 <= f_pos[1] <= 14:
                score += 3
                
            # Đưa chút ngẫu nhiên nhỏ để tránh thiên kiến khi 2 ô bằng điểm
            score += random.random()
                
            if score > best_score:
                best_score = score
                best_frontier_pos = f_pos
                best_frontier_path = f_path
                
        return best_frontier_pos, best_frontier_path
    
    def _bfs_to_target(self, start, target):
        """Dùng BFS để tìm đường đi an toàn khi A* bị kẹt hoặc dùng để Tuần tra"""
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            curr_pos, path = queue.popleft()
            if curr_pos == target:
                return curr_pos, path
                
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (curr_pos[0] + dr, curr_pos[1] + dc)
                if self._is_valid_position(next_pos, self.internal_map) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        return None, []
    
    # Default Helper methods (you can add more)
    
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


class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Ghost Fixed"
        self.last_move = None
        self.prev_positions = []  # lưu lịch sử vị trí để phá loop
        self.guess_enemy_pos = (10, 10)  # trung tâm map
        self.known_map = np.full((21, 21), -1)
        self.last_known_enemy_pos = None
        self.prev_enemy_pos = None
        self.left_bias = 5    # tăng → thích đi trái hơn
        self.right_bias = 15   # tăng → thích đi phải hơn
        self.up_bias = 5     # tăng → thích đi lên hơn  
        self.down_bias = 15 # tăng → thích đi xuống hơn 

        # Cờ đánh dấu: True nghĩa là đang trong giai đoạn khai cuộc mở góc dưới phải
        self.is_opening_phase = True

    def _update_memory(self, map_state):
        # TỐI ƯU: Dùng numpy vectorization thay vì 2 vòng lặp for (21x21)
        # Sẽ thay thế các giá trị -1 trong known_map bằng giá trị thực tế nếu map_state có dữ liệu
        ms_arr = np.array(map_state)
        self.known_map = np.where(ms_arr != -1, ms_arr, self.known_map)

    def _explore(self, pos):
        best_move = Move.STAY
        best_score = -float("inf")

        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)

        for move in moves:
            dx, dy = move.value
            new_pos = (pos[0] + dx, pos[1] + dy)

            if not self._is_valid_position(new_pos):
                continue

            score = 0

            # --- CẬP NHẬT: VÙNG NÉ TRÁNH (SOFT PENALTY) ---
            # Hạn chế đi vào lân cận hàng 13-15, cột 9 (Khu vực Pacman Nhóm 16 bị kẹt)
            if 13 <= new_pos[0] <= 15 and 5 <= new_pos[1] <= 9:
                score -= 70  # Phạt nhẹ: Đủ để né khi đi dạo, nhưng sẽ vượt qua nếu bị ép góc

            # TỐI ƯU MỞ MAP: Phóng tầm mắt xa 4 ô theo hướng di chuyển
            unexplored_count = 0
            check_pos = new_pos
            for _ in range(4):
                if not self._is_valid_position(check_pos):
                    break
                
                # Nếu tia nhìn quét trúng vùng né tránh -> trừ một chút điểm để giảm sự hấp dẫn của sương mù ở đây
                if 13 <= check_pos[0] <= 15 and check_pos[1] == 9:
                    score -= 10
                    
                if self.known_map[check_pos[0]][check_pos[1]] == -1:
                    unexplored_count += 1
                check_pos = (check_pos[0] + dx, check_pos[1] + dy)
            
            # Thưởng điểm khám phá (đã hạ xuống 5 để tránh bệnh nhảy Ping-Pong)
            score += 5 * unexplored_count
            if self.known_map[new_pos[0]][new_pos[1]] == -1:
                score += 10

            escape_routes = self._count_valid_moves(new_pos)
            score += 2 * escape_routes

            if escape_routes <= 1:
                score -= 10

            if self.last_move is not None:
                opposite = {
                    Move.UP: Move.DOWN, Move.DOWN: Move.UP,
                    Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT,
                }
                if move == opposite.get(self.last_move):
                    score -= 500  # Vẫn cấm quay đầu ngay lập tức

            # Hạ án phạt vết chân cũ để không bị treo máy trong ngõ hẹp
            if new_pos in self.prev_positions:
                # Càng đi lại ô gần đây càng bị phạt nặng
                index = self.prev_positions.index(new_pos)
                score -= (80 - index * 10)

            # FIX CHẠY 1 FORM: Cộng bias vào cả lúc đi mở map
            if move == Move.UP:
                score += self.up_bias
            elif move == Move.DOWN:
                score += self.down_bias
            elif move == Move.LEFT:
                score += self.left_bias
            elif move == Move.RIGHT:
                score += self.right_bias

            # Tăng độ ngẫu nhiên thêm một chút để đường đi mềm mại hơn
            score += random.random() * 5

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    # dự đoán hướng đi của ghost để đưa ra hướng đi hợp lí nhất
    def _smart_ghost_escape(self, ghost_pos, pacman_pos):
        dist_map = self._compute_distance_map(pacman_pos)

        best_dist = -1
        best_positions = []

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (ghost_pos[0] + dx, ghost_pos[1] + dy)
            d = dist_map.get(new_pos, -1)
            
            if not self._is_valid_position(new_pos):
                continue
            if self._is_dead(new_pos, pacman_pos):
                continue

            move_vec = (dx, dy)
            pac_vec = (pacman_pos[0] - ghost_pos[0], pacman_pos[1] - ghost_pos[1])
            
            # TỐI ƯU: Thay thế np.sign bằng logic thuần Python (nhanh hơn nhiều khi chạy lặp)
            sign_x = (pac_vec[0] > 0) - (pac_vec[0] < 0)
            sign_y = (pac_vec[1] > 0) - (pac_vec[1] < 0)
            
            if move_vec == (sign_x, sign_y):
                d -= 20

            if d > best_dist:
                best_dist = d
                best_positions = [new_pos]
            elif d == best_dist:
                best_positions.append(new_pos)

        if best_positions:
            return random.choice(best_positions)
        return ghost_pos

    # dự đoán pacman cho ghost huog đi
    def _smart_pacman_move(self, pacman_pos, ghost_pos):
        dist_map = self._compute_distance_map(ghost_pos)

        best_pos = pacman_pos
        best_dist = dist_map.get(pacman_pos, float("inf"))

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (pacman_pos[0] + dx, pacman_pos[1] + dy)

            if not self._is_valid_position(new_pos):
                continue

            d = dist_map.get(new_pos, float("inf"))
            if d < best_dist:
                best_dist = d
                best_pos = new_pos

        return best_pos
    # MONTE CARLO
    def monte_carlo(self, ghost_pos, pacman_pos, map_state, simulations=5):
        total_score = 0

        for _ in range(simulations):
            sim_ghost = ghost_pos
            sim_pacman = pacman_pos
            survived = True

            for step in range(5):
                # FIX CHÓT: XÓA tham số map_state ở đuôi
                sim_ghost = self._smart_ghost_escape(sim_ghost, sim_pacman)

                if self._is_dead(sim_ghost, sim_pacman):
                    total_score += -1000 + step
                    survived = False
                    break

                for _ in range(2):
                    # FIX CHÓT: XÓA tham số map_state ở đuôi
                    sim_pacman = self._smart_pacman_move(sim_pacman, sim_ghost)
                    if self._is_dead(sim_ghost, sim_pacman):
                        total_score += -1000 + step
                        survived = False
                        break

                if not survived:
                    break

            if survived:
                # TỐI ƯU CỰC LỚN: Dùng khoảng cách Manhattan thay vì BFS (giảm tải CPU triệt để)
                dist = abs(sim_ghost[0] - sim_pacman[0]) + abs(sim_ghost[1] - sim_pacman[1])
                total_score += dist

        return total_score / simulations
    #STEP : nếu thấy pac thì chạy ko thì explore
    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state)

        # KIỂM TRA GIAI ĐOẠN KHAI CUỘC
        if self.is_opening_phase:
            # Quét khu vực góc Dưới - Phải (từ hàng 11->20, cột 11->20)
            unexplored_in_corner = 0
            for r in range(11, 21):
                for c in range(11, 21):
                    if self.known_map[r][c] == -1:
                        unexplored_in_corner += 1
            
            # Nếu khu vực này đã mở gần hết (còn sót lại ít hơn 5 ô tối)
            if unexplored_in_corner < 5 or my_position[0] >= 19 or my_position[1] >= 19:
                self.is_opening_phase = False  # Kết thúc khai cuộc
                # Reset bias và tung đồng xu chọn hướng ngẫu nhiên đầu tiên
                self.up_bias = self.down_bias = self.left_bias = self.right_bias = 0
                new_combo = random.choice(["DOWN_RIGHT", "DOWN_LEFT", "UP_RIGHT", "UP_LEFT"])
                if new_combo == "DOWN_RIGHT": self.down_bias = 15; self.right_bias = 15
                elif new_combo == "DOWN_LEFT": self.down_bias = 15; self.left_bias = 15
                elif new_combo == "UP_RIGHT": self.up_bias = 15; self.right_bias = 15
                elif new_combo == "UP_LEFT": self.up_bias = 15; self.left_bias = 15


        # NẾU ĐÃ XONG KHAI CUỘC -> CHUYỂN SANG ĐÁNH RANDOM PHÁ FORM (4%)
        else:
            if random.random() < 0.04:
                self.up_bias = self.down_bias = self.left_bias = self.right_bias = 0
                new_combo = random.choice(["DOWN_RIGHT", "DOWN_LEFT", "UP_RIGHT", "UP_LEFT"])
                if new_combo == "DOWN_RIGHT": self.down_bias = 15; self.right_bias = 15
                elif new_combo == "DOWN_LEFT": self.down_bias = 15; self.left_bias = 15
                elif new_combo == "UP_RIGHT": self.up_bias = 15; self.right_bias = 15
                elif new_combo == "UP_LEFT": self.up_bias = 15; self.left_bias = 15

        # Cập nhật trí nhớ vị trí kẻ thù
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        elif self.last_known_enemy_pos is not None:
            enemy_position = self.last_known_enemy_pos

        # Xử lý vòng lặp vị trí (chống kẹt)
        self.prev_positions.append(my_position)
        if len(self.prev_positions) > 6:
            self.prev_positions.pop(0)

        # Tính toán hướng đi của kẻ thù
        enemy_dir = None
        if enemy_position is not None and self.prev_enemy_pos is not None:
            enemy_dir = (
                enemy_position[0] - self.prev_enemy_pos[0],
                enemy_position[1] - self.prev_enemy_pos[1]
            )

        self.prev_enemy_pos = enemy_position

        # Nếu không thấy địch, chuyển sang mode khám phá
        if enemy_position is None:
            return self._explore(my_position)

        threat = enemy_position
        predicted_threat = threat

        if enemy_dir is not None:
            predicted_threat = (
                threat[0] + enemy_dir[0] * 2,
                threat[1] + enemy_dir[1] * 2
            )
            if not self._is_valid_position(predicted_threat):
                predicted_threat = threat

        best_move = Move.STAY
        best_score = -float("inf")
        best_moves = []

        dist_map = self._compute_distance_map(predicted_threat)

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            if not self._is_valid_position(new_pos):
                continue

            # TỐI ƯU: Gộp điều kiện _is_dead (tránh gọi hàm này tới 2 lần như code cũ)
            if self._is_dead(new_pos, threat):
                continue

            score = 0
            dx_p = threat[0] - my_position[0]
            dy_p = threat[1] - my_position[1]

            if (dx_p != 0 and dx == int(dx_p / abs(dx_p))) or \
               (dy_p != 0 and dy == int(dy_p / abs(dy_p))):
                score -= 150

            escape_routes = self._count_valid_moves(new_pos)
            score += 2 * escape_routes

            mc_score = self.monte_carlo(new_pos, enemy_position, map_state, simulations=3)
            score += 0.5 * mc_score
            #cho điểm bia để phá lặp form
            if move == Move.UP: 
                score += 3 * self.up_bias
            if move == Move.DOWN: 
                score += 3 * self.down_bias
            if move == Move.LEFT: 
                score += 3 * self.left_bias
            if move == Move.RIGHT: 
                score += 3 * self.right_bias

            # Phạt biên
            if new_pos[0] == 0 or new_pos[0] == 20 or new_pos[1] == 0 or new_pos[1] == 20:
                score -= 80
            if new_pos[0] <= 1 or new_pos[0] >= 19 or new_pos[1] <= 1 or new_pos[1] >= 19:
                score -= 40

            if escape_routes <= 1:
                score -= 100

            dist = dist_map.get(new_pos, -1)
            if dist == -1:
                continue

            if self.known_map[new_pos[0]][new_pos[1]] == -1:
                score -= 10

            direct_dist = abs(new_pos[0] - predicted_threat[0]) + abs(new_pos[1] - predicted_threat[1])

            if enemy_dir is not None:
                move_vec = (dx, dy)
                if move_vec == enemy_dir:
                    score -= 100
                if move_vec == (-enemy_dir[0], -enemy_dir[1]):
                    score += 50

            future_escape = 0
            for move2 in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx2, dy2 = move2.value
                next2 = (new_pos[0] + dx2, new_pos[1] + dy2)
                if self._is_valid_position(next2):
                    future_escape += 1

            score += 1.5 * future_escape
            score += 2 * dist
            score += 2 * direct_dist

            if dist <= 3:
                score -= 400
            elif dist <= 5:
                score -= 100

            if escape_routes <= 1:
                if dist > 5:
                    score += 10
                else:
                    score -= 30

            if self.last_move is not None:
                opposite = {
                    Move.UP: Move.DOWN, Move.DOWN: Move.UP,
                    Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT,
                }
                if move == opposite.get(self.last_move):
                    score -= 500  # Vẫn cấm quay đầu ngay lập tức

            # Hạ án phạt vết chân cũ để không bị treo máy trong ngõ hẹp
            if new_pos in self.prev_positions:
                index = self.prev_positions.index(new_pos)
                score -= (80 - index * 10)

            score += random.random()

            if score > best_score:
                best_score = score
                best_moves = [move]
            elif abs(score - best_score) < 1e-6:
                best_moves.append(move)

        if best_moves:
            best_move = random.choice(best_moves)

        # Fallback nếu điểm quá tệ
        if best_score < 20:
            fallback_move = None
            max_dist = -1
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dx, dy = move.value
                new_pos = (my_position[0] + dx, my_position[1] + dy)
                if not self._is_valid_position(new_pos):
                    continue
                d = abs(new_pos[0] - threat[0]) + abs(new_pos[1] - threat[1])
                if d > max_dist:
                    max_dist = d
                    fallback_move = move
            if fallback_move is not None:
                best_move = fallback_move

        self.last_move = best_move

        if best_move == Move.STAY:
            return self._random_move(my_position)

        return best_move


    # cho phép đi vào vùng ko biết
    def _is_valid_position(self, pos):
        row, col = pos
        return (
            0 <= row < 21 and
            0 <= col < 21 and
            self.known_map[row][col] != 1   # chỉ cấm tường
        )

    #bfs từ vị trí start_pos đề tính khoảng cách
    def _compute_distance_map(self, start_pos, map_state=None):
        # FIX LỚN: Bỏ hoàn toàn logic dùng map_state.shape để chống crash game
        # Chỉ sử dụng _is_valid_position để tự do đi vào vùng sương mù (-1)
        def valid(pos):
            return self._is_valid_position(pos)

        queue = deque([(start_pos, 0)])
        visited = {start_pos: 0}

        while queue:
            (row, col), dist = queue.popleft()

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                nr, nc = row + dr, col + dc

                if (nr, nc) not in visited and valid((nr, nc)):
                    visited[(nr, nc)] = dist + 1
                    queue.append(((nr, nc), dist + 1))

        return visited
    # khi ko bt pac đâu sẽ cho random hướng đi
    def _random_move(self, my_position):
        moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(moves)

        for move in moves:
            dx, dy = move.value
            new_pos = (my_position[0] + dx, my_position[1] + dy)

            if self._is_valid_position(new_pos):
                return move

        return Move.STAY

    #đếm số đường đi hợp lệ từ 1 vị trí
    def _count_valid_moves(self, pos):
        count = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dx, dy = move.value
            new_pos = (pos[0] + dx, pos[1] + dy)

            if self._is_valid_position(new_pos):
                count += 1

        return count

    def _max_valid_steps(self, pos, move, max_steps):
        steps = 0
        current = pos

        for _ in range(max_steps):
            dx, dy = move.value
            next_pos = (current[0] + dx, current[1] + dy)

            if not self._is_valid_position(next_pos):
                break

            steps += 1
            current = next_pos

        return steps
    def _is_dead(self, ghost_pos, pacman_pos):
        if pacman_pos is None:
            return False
        return abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) <= 1
