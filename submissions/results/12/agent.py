"""
Combined Advanced learning Seeker and Evasive Ghost for Limited Vision.
"""

import sys
import random
import itertools
from pathlib import Path
from collections import deque
from heapq import heappush, heappop
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    Solution: Kết hợp Memory mapping với Q-learning prediction.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Learning Seeker (Limited Vision)"
        self.current_path = []
        
        # --- BỘ NHỚ LIMITED VISION ---
        self.memory_map = None
        self.last_known_enemy_pos = None

        # --- THEO DÕI VỊ TRÍ GHOST ĐỂ TÍNH ĐÀ ---
        self.last_enemy_pos = None
        self.prev_enemy_pos = None
        self.last_momentum = None
        
        # --- HỌC TĂNG CƯỜNG (Q-Learning) CHO NGƯỠNG KHOẢNG CÁCH ---
        # Tự động học giá trị ngưỡng tối ưu khi Pacman lại gần Ghost
        self.threshold_options = [2, 3, 4, 5, 6, 7]
        self.q_values = {t: 0.0 for t in self.threshold_options} # Điểm số cho mỗi ngưỡng
        self.current_threshold = 4 # Bắt đầu với ngưỡng là 4
        self.steps_with_threshold = 0
        self.start_distance = None
        self.exploration_rate = 0.3 # 30% tỷ lệ chọn ngẫu nhiên (explore)

    def _manhattan_distance(self, pos1, pos2):
        """Tính toán khoảng cách Manhattan giữa 2 điểm."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    # SỬ DỤNG MEMORY MAP thay cho map_state
    def _get_maze_distances(self, start):
        """Dùng BFS để tìm khoảng cách thực tế trong memory map."""
        if self.memory_map is None: return {}
        queue = deque([(start, 0)])
        visited = {start: 0}
        while queue:
            current, dist = queue.popleft()
            for next_pos, _ in self._get_neighbors(current, self.memory_map):
                if next_pos not in visited:
                    visited[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))
        return visited

    def predict_enemy_move(self, enemy_pos, my_pos, distances_from_me):
        """Dự đoán vị trí tiếp theo qua memory map dựa trên hướng đi hiện tại và khoảng cách tới Pacman."""
        neighbors = self._get_neighbors(enemy_pos, self.memory_map)
        if not neighbors:
            return enemy_pos
        
        # Tính hướng đi gần đây của Ghost
        momentum_dir = None
        if self.prev_enemy_pos is not None and self.prev_enemy_pos != enemy_pos:
            momentum_dir = (enemy_pos[0] - self.prev_enemy_pos[0], 
                           enemy_pos[1] - self.prev_enemy_pos[1])
        
        best_move = Move.STAY
        best_score = -1
        
        for next_pos, move in neighbors:
            # Ưu tiên các hướng của Ghost nếu chúng là chưa biết (-1) hoặc đã biết (0)
            # khoảng cách BFS từ Ghost đến Pacman
            dist = distances_from_me.get(next_pos, self._manhattan_distance(next_pos, my_pos))
            score = dist
            
            # Thêm điểm nếu Ghost tiếp tục đi theo hướng cũ (không quay đầu)
            if momentum_dir is not None:
                move_dir = move.value
                if move_dir == momentum_dir:
                    score += 2 # Thưởng lớn nếu Ghost duy trì hướng đi
            
            if score > best_score:
                best_score = score
                best_move = move
                
        # Giả lập vị trí di chuyển mong muốn
        return (enemy_pos[0] + best_move.value[0], enemy_pos[1] + best_move.value[1])
        
    def astar(self, start, goal):
        """A* trên memory map. Mục tiêu có thể là ô chưa biết (-1)."""
        if start == goal: return [Move.STAY]

        def heuristic(pos):
            return self._manhattan_distance(pos, goal)
        
        counter = itertools.count()
        frontier = [(heuristic(start), next(counter), start)]
        came_from = {start: (None, None)}
        g_cost = {start: 0}
        
        while frontier:
            _, _, current = heappop(frontier)
            
            if current == goal:
                path = []
                while current != start:
                    parent, move = came_from[current]
                    path.append(move)
                    current = parent
                path.reverse()
                return path if path else [Move.STAY]
            
            for next_pos, move in self._get_neighbors(current, self.memory_map):
                # Không đi xuyên sương mù nếu đích không phải sương mù đó
                if current != start and self.memory_map[next_pos] == -1 and next_pos != goal:
                    continue
                    
                new_cost = g_cost[current] + 1
                if next_pos not in g_cost or new_cost < g_cost[next_pos]:
                    g_cost[next_pos] = new_cost
                    f_cost = new_cost + heuristic(next_pos)
                    came_from[next_pos] = (current, move)
                    heappush(frontier, (f_cost, next(counter), next_pos))
        
        return []

    def explore_bfs(self, start):
        """Khám phá ô sương mù (-1) thông minh bằng cách chấm điểm thể tích sương mù."""
        from collections import deque
        queue = deque([(start, [])])
        visited = set([start])
        
        height, width = self.memory_map.shape
        frontiers = []
        
        # Tìm các ô sương mù gần nhất
        while queue:
            current, path = queue.popleft()
            # Nếu gặp ô sương mù thì thêm vào danh sách
            if self.memory_map[current] == -1:
                frontiers.append((current, path))
                continue
            
            for next_pos, move in self._get_neighbors(current, self.memory_map):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
                    
        if not frontiers:
            # Bản đồ đã mở hết nhưng chưa thấy Ghost -> Tuần tra các "góc khuất"
            empty_cells = np.argwhere(self.memory_map == 0)
            if len(empty_cells) > 0:
                corners = []
                for cell in empty_cells:
                    pos = tuple(cell)
                    # Góc khuất/ngõ cụt thường chỉ có 1 hoặc 2 hướng đi
                    if len(self._get_neighbors(pos, self.memory_map)) <= 2:
                        corners.append(pos)
                
                if corners:
                    # Ưu tiên các góc khuất ở ngay sát (<= 3 bước) để dọn sạch trước
                    np.random.shuffle(corners)
                    
                    # Kiểm tra góc khuất gần trước
                    for target_cell in corners:
                        if start != target_cell and self._manhattan_distance(start, target_cell) <= 3:
                            path_to_target = self.astar(start, target_cell)
                            if path_to_target and path_to_target != [Move.STAY]:
                                return path_to_target
                                
                    # Nếu không có góc khuất gần, chọn ngẫu nhiên một góc khuất xa để tuần tra
                    for target_cell in corners:
                        if self._manhattan_distance(start, target_cell) > 3:
                            path_to_target = self.astar(start, target_cell)
                            if path_to_target and path_to_target != [Move.STAY]:
                                return path_to_target
                                
                # Chọn ngẫu nhiên ô trống xa bất kỳ để đi đến
                np.random.shuffle(empty_cells)
                for cell in empty_cells:
                    target_cell = tuple(cell)
                    if self._manhattan_distance(start, target_cell) > 3:
                        path_to_target = self.astar(start, target_cell)
                        if path_to_target and path_to_target != [Move.STAY]:
                            return path_to_target
                            
                return self.astar(start, tuple(empty_cells[0]))
            return []
            
        best_path = []
        best_score = float('-inf')
        
        for pos, path in frontiers:
            r, c = pos
            r_min = max(0, r - 2)
            r_max = min(height, r + 3)
            c_min = max(0, c - 2)
            c_max = min(width, c + 3)
            
            window = self.memory_map[r_min:r_max, c_min:c_max]
            fog_volume = np.sum(window == -1)
            
            # Khối lượng sương mù x 2.0 điểm trừ đi Khoảng cách bước
            score = (fog_volume * 2.0) - len(path)
            
            # Ưu tiên dọn dẹp các ngách/góc khuất (sương mù) nếu nó ở quá gần (<= 3 bước)
            # Tránh việc bỏ sót các góc trên bản đồ nơi Ghost có thể đang nấp!
            if len(path) <= 3:
                score += 100
            
            if score > best_score:
                best_score = score
                best_path = path
                
        return best_path

    def _get_neighbors(self, pos, map_source):
        """Lấy danh sách điểm có thể đi (tránh tường số 1)."""
        height, width = map_source.shape
        neighbors = []
        moves = [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]
        # Random thứ tự di chuyển để tránh bị kẹt ở một hướng
        random.shuffle(moves)
        for move in moves:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < height and 0 <= nc < width:
                if map_source[nr, nc] != 1: 
                    neighbors.append(((nr, nc), move))
        return neighbors

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # --- BƯỚC 1: CẬP NHẬT TRÍ NHỚ TỪ TẦM NHÌN HẠN CHẾ ---
        if self.memory_map is None:
            self.memory_map = np.full(map_state.shape, -1)
        
        visible_mask = map_state != -1
        self.memory_map[visible_mask] = map_state[visible_mask]
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            # Có tầm nhìn, update momentum chính xác
            self.prev_enemy_pos = getattr(self, 'last_enemy_pos', None)
            self.last_enemy_pos = enemy_position
        elif self.last_known_enemy_pos == my_position:
            # Mất dấu Enemy, và ta đã tới chỗ đó nhưng không thấy
            self.last_known_enemy_pos = None
            self.prev_enemy_pos = None
            self.last_enemy_pos = None
            
        target = enemy_position or self.last_known_enemy_pos
        
        current_momentum = None
        if self.prev_enemy_pos is not None and getattr(self, 'last_enemy_pos', None) is not None and self.prev_enemy_pos != self.last_enemy_pos:
            current_momentum = (self.last_enemy_pos[0] - self.prev_enemy_pos[0],
                                self.last_enemy_pos[1] - self.prev_enemy_pos[1])
                                
        momentum_changed = False
        if getattr(self, 'last_momentum', None) is not None and current_momentum is not None:
            if current_momentum != self.last_momentum and current_momentum != (0, 0):
                momentum_changed = True
        self.last_momentum = current_momentum

        # KIỂM TRA ĐƯỜNG ĐI ĐÃ LƯU
        # Xem sương mù mới mở có lộ ra cái TƯỜNG (1) chặn current_path không
        path_is_blocked = False
        if self.current_path:
            sim_pos = my_position
            for m in self.current_path:
                sim_pos = (sim_pos[0] + m.value[0], sim_pos[1] + m.value[1])
                # Check for boundary conditions safely just in case
                if 0 <= sim_pos[0] < self.memory_map.shape[0] and 0 <= sim_pos[1] < self.memory_map.shape[1]:
                    if self.memory_map[sim_pos] == 1:
                        path_is_blocked = True
                        break
                else:
                    path_is_blocked = True
                    break

        # --- BƯỚC 2: QUYẾT ĐỊNH TARGET VÀ SEARCH ĐƯỜNG ĐI ---
        should_replan = True
        
        if target is None:
            # HOÀN TOÀN MẤT DẤU -> CHẾ ĐỘ THĂM DÒ TÌM KIẾM (EXPLORATION)
            if not self.current_path or path_is_blocked:
                self.current_path = self.explore_bfs(my_position)
            should_replan = False
        else:
            # BIẾT VỊ TRÍ GHOST -> HỌC TĂNG CƯỜNG VÀ A*
            maze_distances = self._get_maze_distances(my_position)
            dist_to_enemy = maze_distances.get(target, self._manhattan_distance(my_position, target))
            
            if self.start_distance is None:
                self.start_distance = dist_to_enemy
            
            self.steps_with_threshold += 1
            # Đánh giá kết quả của 'ngưỡng' hiện tại sau mỗi 7 bước
            if self.steps_with_threshold >= 7:
                distance_closed = self.start_distance - dist_to_enemy
                reward = distance_closed # Reward dương nếu rút ngắn khoảng cách, âm nếu bị bỏ xa
                alpha = 0.5 # Tốc độ học (learning rate)
                self.q_values[self.current_threshold] = (1 - alpha) * self.q_values[self.current_threshold] + alpha * reward
                
                # Chọn ngưỡng tiếp theo
                if random.random() < self.exploration_rate: 
                    self.current_threshold = random.choice(self.threshold_options) # Random threshold
                else:
                    self.current_threshold = max(self.threshold_options, key=lambda t: self.q_values[t]) # Chọn threshold tốt nhất
                    
                self.steps_with_threshold = 0
                self.start_distance = dist_to_enemy

            # Tính toán việc lập lại đường đi nếu Ghost đã đi chệch hoặc vào gần ngưỡng
            should_replan = (not self.current_path or 
                             path_is_blocked or
                             self.prev_enemy_pos is None or
                             self._manhattan_distance(target, self.prev_enemy_pos) > 1 or
                             dist_to_enemy <= self.current_threshold or
                             momentum_changed) # Tính lại đường đi ngay khi Ghost đổi hướng
                             
            if should_replan:
                if dist_to_enemy <= 2:
                    # Nếu cực sát (<= 2 ô), lao thẳng tới tấn công Ghost
                    target_pos = target
                else:
                    # Khi ở xa, dự đoán trước vị trí Ghost đến để chạy đón đầu
                    target_pos = self.predict_enemy_move(target, my_position, maze_distances)

                self.current_path = self.astar(my_position, target_pos)
                if not self.current_path: 
                    self.current_path = self.astar(my_position, target)

        # --- BƯỚC 3: DI CHUYỂN BẰNG ĐƯỜNG ĐÃ DỰ TÍNH VÀ CỘNG DỒN MOVE ---
        if self.current_path and self.current_path[0] != Move.STAY:
            move = self.current_path[0]
            consecutive_steps = 0
            
            for m in self.current_path:
                if m == move and consecutive_steps < self.pacman_speed:
                    consecutive_steps += 1
                else:
                    break
            
            # Đảm bảo Pacman không tự giới hạn bản thân (giữ Speed-2 kể cả khi ô đích là 1 ô)
            if len(self.current_path) == consecutive_steps and consecutive_steps < self.pacman_speed:
                consecutive_steps = self.pacman_speed
                
            actual_steps = self._max_valid_steps(my_position, move, map_state, consecutive_steps)
            if actual_steps > 0:
                self.current_path = self.current_path[actual_steps:]
                return (move, actual_steps)
            else:
                self.current_path = []
        
        # Mặc định random nếu kẹt đường
        random_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(random_moves)
        for move in random_moves:
            if self._is_valid_move(my_position, move, map_state):
                 return (move, 1)
                 
        return (Move.STAY, 1)

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
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] != 1


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = (15, 10)
        #self.enemy_possible_position = []
        self.my_previous_move = Move.STAY
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Pacman's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        self._update_enemy_position(enemy_position)
        best_choice = self.MCTS(my_position, map_state, step_number)
        self.my_previous_move = best_choice
        return best_choice
    
    def _update_enemy_position(self, enemy_position):
        if (enemy_position == None): return
        self.last_known_enemy_pos = enemy_position
        self.enemy_possible_position = []

    def MCTS(self, my_position, map_state, step_number):
        import math 

        class moveNode:
            def __init__(self, move, position):
                self.move = move
                self.position = position
                self.value = 0
                self.child_nodes = []
                self.total_simulation = 0
                self.total_win = 0

            def _get_value(self):
                if (self.total_simulation == 0):
                    return float('+inf')
                return self.total_win / self.total_simulation

            def _get_childs(self):
                if (self.child_nodes == []): return None
                return self.child_nodes
            
            def _generate_childs(self, possible_moves):
                for next_pos, move in possible_moves:
                    self.child_nodes.append(moveNode(move, next_pos))
                return self.child_nodes
            
            def _update_stats(self, value):
                self.total_win += value
                self.total_simulation += 1
        
        def _selection(root):
            selection_chain = [ root ]
            parent_node = root
            child_list = parent_node._get_childs()
            while (child_list is not None and len(child_list) > 0):
                best_node = None
                best_score = float('-inf')
                for child in child_list:
                    child_value = child._get_value()
                    if (child_value == float('+inf')): UCB1 = child_value
                    else: UCB1 = child._get_value() + math.sqrt(2 * math.log(parent_node.total_simulation) / child.total_simulation)
                    if (UCB1 > best_score):
                        best_node = child
                        best_score = UCB1
                child_list = best_node._get_childs()
                selection_chain.append(best_node)
            return selection_chain

        def _simulate(node, step_number):
            import random

            def _pick_enemy_position():
                return self.last_known_enemy_pos

            def _simulate_enemy(my_position, enemy_position):
                import random
                best_score = float('+inf')
                best_next_positions = []
                for new_position, _ in self._new_get_neighbors(enemy_position, map_state):
                   current_score = self._manhattan_distance(new_position, my_position)
                   if (best_score > current_score):
                       best_next_positions = [new_position]
                       best_score = current_score
                   elif (best_score == current_score):
                       best_next_positions.append(new_position)
                if random.random() < 0.1:
                    neighbors = self._new_get_neighbors(enemy_position, map_state)
                    if neighbors:
                        return random.choice(neighbors)[0]
                return random.choice(best_next_positions) if best_next_positions else None
            
            def _simulate_ghost(my_position, enemy_position):
                import random
                best_score = float('-inf')
                best_next_positions = []
                for new_position, _ in self._new_get_neighbors(my_position, map_state):
                   current_score = self._manhattan_distance(new_position, enemy_position)
                   if (best_score < current_score):
                       best_next_positions = [new_position]
                       best_score = current_score
                   elif (best_score == current_score):
                       best_next_positions.append(new_position)
                if random.random() < 0.3:
                    neighbors = self._new_get_neighbors(my_position, map_state)
                    if neighbors:
                        return random.choice(neighbors)[0]
                return random.choice(best_next_positions) if best_next_positions else None
            
            my_current_position = node.position
            current_enemy_position = _pick_enemy_position()
            
            while (step_number < 200):
                enemy_next_position = _simulate_enemy(my_current_position, current_enemy_position)
                my_next_position = _simulate_ghost(my_current_position, current_enemy_position)
                
                if enemy_next_position is None or my_next_position is None:
                    break
                if (self._manhattan_distance(enemy_next_position, my_next_position) < 2): 
                    return step_number
                step_number += 1
                current_enemy_position = enemy_next_position
                my_current_position = my_next_position
            return step_number

        def _back_propagation(selection_chain, simulation_result):
            for node in selection_chain:
                node._update_stats(simulation_result)
                
        root = moveNode(None, my_position)
        while (root.total_simulation < 100):
            
            selection_chain = _selection(root)
            node = selection_chain[-1]
            child_nodes = []
            child_nodes = node._generate_childs(self._new_get_neighbors(node.position, map_state))
            for child in child_nodes:
                
                simulation_result = _simulate(child, step_number + len(selection_chain) - 1)
                new_chain = selection_chain + [child]
                _back_propagation(new_chain, simulation_result)
        
        best_score = float('-inf')
        best_choice = None
        for node in root.child_nodes:
            current_score = node._get_value()
            if (current_score == best_score and node.move == self._get_reverse_move(self.my_previous_move)):
                if (random.random() < 0.8): continue
            if (current_score > best_score):
                best_choice = node
                best_score = current_score
        
        if best_choice is None:
            return Move.STAY
        return best_choice.move

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []
        allMove = [ Move.UP, Move.RIGHT, Move.LEFT, Move.DOWN]
        for move in allMove:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors
    
    def _get_pacman_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []
        allMove = [  Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT ]
        for move in allMove:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors

    def _get_enemy_neighbors(self, enemy_pos, map_state):
        neighbors = self._get_pacman_neighbors(enemy_pos, map_state)
        new_neighbors = neighbors
        for new_pos, move in neighbors:
            new_pos = self._apply_move(new_pos, move)
            if self._is_valid_position(new_pos, map_state):
                new_neighbors.append((new_pos, move))
        return new_neighbors
    
    def _manhattan_distance(self, pos1, pos2):
            """Calculate Manhattan distance between two positions."""
            return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
            # Helper methods (you can add more)
    
    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)
    
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
    
    def _get_reverse_move(self, move):
        if (move == Move.STAY): return move
        move_list = { Move.DOWN : Move.UP,
                      Move.UP : Move.DOWN,
                      Move.RIGHT : Move.LEFT,
                      Move.LEFT : Move.RIGHT}
        return move_list[move]

    def _new_get_neighbors(self, position, map_state):
        """Lấy danh sách điểm có thể đi (tránh tường số 1)."""
        height, width = map_state.shape
        neighbors = []
        moves = [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]

        for move in moves:
            nr, nc = position[0] + move.value[0], position[1] + move.value[1]
            if 0 <= nr < height and 0 <= nc < width:
                if map_state[nr, nc] != 1: 
                    neighbors.append(((nr, nc), move))
        return neighbors