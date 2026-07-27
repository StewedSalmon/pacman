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
from collections import deque
import numpy as np
import random
import heapq


class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Smart Explorer Pacman"

        # --- Map memory + heatmap ---
        self.memory_map = np.full((21, 21), -1, dtype=int)
        self.heatmap    = np.zeros((21, 21), dtype=int)

        # --- Committed path (tránh tính lại mỗi bước) ---
        self.current_path   = []   # list[(row,col)] đang đi
        self.current_target = None # frontier đang nhắm

        # --- Memory ghost ---
        self.last_known_enemy_pos = None

        # --- A* chase path cache ---
        self.path = []  # Store planned path: list[Move]

    # ================================================================
    # MAIN STEP
    # ================================================================
    def step(self, map_state, my_position, enemy_position, step_number):
        
        # In tầm nhìn ra console để debug
        self._print_vision(map_state, my_position, enemy_position)

        # 1. Heatmap
        self.heatmap[my_position[0], my_position[1]] += 1

        # 2. Cập nhật memory_map
        visible = map_state != -1
        self.memory_map[visible] = map_state[visible]

        # 3. Cập nhật memory ghost — lưu vị trí CŨ trước khi update
        prev_enemy_pos = self.last_known_enemy_pos
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        # 4. Tới chỗ cũ của ghost mà không thấy → xóa
        if target is not None and enemy_position is None:
            if my_position == target:
                self.last_known_enemy_pos = None
                self.path = []  # Xóa path cũ luôn
                target = None

        # --- SWITCH MODE ---
        if target is None:
            self.path = []  # Reset path khi không có target
            return self._explore_smart(my_position)
        else:
            self.current_path   = []  # Reset exploration path
            self.current_target = None
            return self._chase_astar(my_position, target, prev_enemy_pos)

    # ================================================================
    # CHASE bằng A*
    # ================================================================
    def _chase_astar(self, my_position, target, prev_enemy_pos):
        # Tính lại A* nếu: path rỗng HOẶC địch vừa di chuyển sang vị trí mới
        if not self.path or target != prev_enemy_pos:
            self.path = self.astar(my_position, target)

        if not self.path:
            return self._fallback(my_position)

        # Đếm số bước đi thẳng liên tiếp
        first_move = self.path[0]
        consecutive_steps = 0
        for move in self.path:
            if move == first_move:
                consecutive_steps += 1
            else:
                break

        steps_to_take = min(consecutive_steps, self.pacman_speed)

        # Xóa các bước đã đi khỏi path cache
        self.path = self.path[steps_to_take:]

        return (first_move, steps_to_take)

    # ================================================================
    # A* SEARCH
    # ================================================================
    def astar(self, start, goal):
        frontier = []
        heapq.heappush(frontier, (0, 0, start, [], None))
        visited = {start: 0}

        while frontier:
            priority, cost, current, path, last_move = heapq.heappop(frontier)

            if current == goal:
                return path

            for next_pos, move in self._get_neighbors(current):
                # Phạt rẽ ngoặt vì đi thẳng có thể đi nhiều bước/lượt
                turn_penalty = 1 if (last_move is None or move == last_move) else 2
                new_cost = cost + turn_penalty

                if next_pos not in visited or new_cost < visited[next_pos]:
                    visited[next_pos] = new_cost
                    new_priority = new_cost + self.heuristic(next_pos, goal)
                    heapq.heappush(frontier, (new_priority, new_cost, next_pos, path + [move], move))

        return []

    def heuristic(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _get_neighbors(self, pos):
        neighbors = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            next_p = (pos[0] + dr, pos[1] + dc)
            if self._inbounds(*next_p) and self.memory_map[next_p[0], next_p[1]] != 1:
                neighbors.append((next_p, move))
        return neighbors

    # ================================================================
    # EXPLORATION THÔNG MINH
    # ================================================================
    def _explore_smart(self, my_position):
        """
        Đi theo committed path đến frontier.
        Chỉ tính lại path khi: hết path, path bị block, hoặc target bị visited rồi.
        """
        # Kiểm tra path hiện tại còn hợp lệ không
        if self.current_path:
            # Bỏ qua các bước đã ở đó rồi
            while self.current_path and self.current_path[0] == my_position:
                self.current_path.pop(0)

            if self.current_path:
                next_pos = self.current_path[0]
                # Kiểm tra ô kế tiếp có bị tường không (vừa discover)
                if self.memory_map[next_pos[0], next_pos[1]] == 1:  
                    # Path bị block → tính lại
                    self.current_path   = []
                    self.current_target = None
                else:
                    self.current_path.pop(0)
                    return self._move_to(my_position, next_pos)

        # Hết path hoặc bị block → tìm frontier mới
        frontier = self._best_frontier(my_position)

        if frontier is None:
            # Không còn frontier → đi ngẫu nhiên
            return self._fallback(my_position)

        self.current_target = frontier
        self.current_path   = self._bfs_path(my_position, frontier)

        if self.current_path:
            next_pos = self.current_path.pop(0)
            return self._move_to(my_position, next_pos)

        return self._fallback(my_position)

    def _best_frontier(self, start):
        """
        BFS từ start, tìm tất cả frontier trong vùng reachable.
        Frontier = ô memory_map==0, chưa visited nhiều (heatmap thấp),
                   có ít nhất 1 ô kề là -1.
        Chọn frontier có score = distance + heatmap_penalty thấp nhất.
        """
        queue    = deque([(start, 0)])
        seen     = {start}
        candidates = []   # list[(score, frontier_pos)]

        while queue:
            pos, dist = queue.popleft()
            row, col  = pos

            if self._is_frontier(pos):
                # Score = khoảng cách + penalty nếu đã đi qua nhiều
                heat_penalty = self.heatmap[row, col] * 2
                candidates.append((dist + heat_penalty, pos))
                # Không cần BFS sâu hơn từ ô này nữa
                continue

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = row + dr, col + dc
                npos   = (nr, nc)
                if npos not in seen and self._inbounds(nr, nc):
                    # Chỉ đi qua ô đã biết là trống
                    if self.memory_map[nr, nc] == 0:
                        seen.add(npos)
                        queue.append((npos, dist + 1))

        if not candidates:
            return None

        # Chọn frontier có score thấp nhất (gần + ít đi lại)
        candidates.sort(key=lambda x: x[0])
        return candidates[0][1]

    def _is_frontier(self, pos):
        """
        Ô là frontier nếu:
        - memory_map == 0 (biết là trống)
        - Có ít nhất 1 ô kề là -1 (chưa biết)
        """
        row, col = pos
        if self.memory_map[row, col] != 0:
            return False

        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = row + dr, col + dc
            if self._inbounds(nr, nc) and self.memory_map[nr, nc] == -1:
                return True
        return False

    def _bfs_path(self, start, goal):
        """
        BFS tìm đường ngắn nhất từ start → goal trên memory_map.
        Chỉ đi qua ô == 0.
        Trả về list[(row,col)] không gồm start, có gồm goal.
        """
        if start == goal:
            return []

        queue = deque([(start, [])])
        seen  = {start}

        while queue:
            pos, path = queue.popleft()
            row, col  = pos

            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = row + dr, col + dc
                npos   = (nr, nc)

                if npos in seen or not self._inbounds(nr, nc):
                    continue
                if self.memory_map[nr, nc] == 1: # Chỉ né tường
                    continue

                new_path = path + [npos]

                if npos == goal:
                    return new_path

                seen.add(npos)
                queue.append((npos, new_path))

        return []  # Không có đường

    # ================================================================
    # HELPERS
    # ================================================================
    def _move_to(self, from_pos, to_pos):
        """Chuyển 2 ô liền kề thành (Move, steps)."""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]

        if dr == 1:    move = Move.DOWN
        elif dr == -1: move = Move.UP
        elif dc == 1:  move = Move.RIGHT
        else:          move = Move.LEFT

        # Dùng memory_map để tận dụng speed nếu đường thẳng trống
        steps = self._max_valid_steps(from_pos, move, self.memory_map, self.pacman_speed)
        return (move, max(1, steps))

    def _fallback(self, pos):
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            steps = self._max_valid_steps(pos, move, self.memory_map, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def _max_valid_steps(self, pos, move, map_ref, max_steps):
        steps, cur = 0, pos
        for _ in range(max(1, max_steps)):
            dr, dc  = move.value
            npos    = (cur[0] + dr, cur[1] + dc)
            if not self._inbounds(*npos):
                break
            # CẬP NHẬT: Chỉ break khi đụng tường (1). Các ô 0 và -1 đều được đi.
            if map_ref[npos[0], npos[1]] == 1: 
                break
            steps += 1
            cur    = npos
        return min(steps, max_steps) # Đảm bảo không bao giờ vượt qua max_steps

    def _inbounds(self, row, col):
        return 0 <= row < 21 and 0 <= col < 21

    def _desired_steps(self, move, row_diff, col_diff):
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        return abs(col_diff)
    
    def _print_vision(self, map_state, my_position, enemy_position):
        print("\n" + "="*40)
        print(f"TẦM NHÌN TẠI BƯỚC NÀY - Vị trí: {my_position}")
        print("="*40)
        
        for r in range(21):
            row_str = ""
            for c in range(21):
                pos = (r, c)
                
                # Ưu tiên in nhân vật trước
                if pos == my_position:
                    row_str += " P "  # P: Pacman
                elif enemy_position is not None and pos == enemy_position:
                    row_str += " G "  # G: Ghost (nếu nhìn thấy)
                else:
                    cell = map_state[r, c]
                    if cell == -1:
                        row_str += "   "  # Vùng tối: In khoảng trắng (không nhìn thấy)
                    elif cell == 1:
                        row_str += "███"  # Tường: In khối vuông đặc
                    elif cell == 0:
                        row_str += " . "  # Đường đi trống
                    else:
                        # Dành cho các object khác (vd: đồ ăn, điểm số...)
                        row_str += f" {cell} " 
                        
            # Chỉ in ra những dòng có chứa thông tin (có đường, có tường hoặc nhân vật)
            # Bỏ qua các dòng trống hoàn toàn (toàn vùng tối) để console đỡ dài
            if row_str.strip() != "":
                print(row_str)
                
        print("="*40 + "\n")
    

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ghost_2"
        self.last_known_enemy_pos = None
        self.last_pos = None # Lưu vị trí ngay trước đó để cấm quay đầu

    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        # --- PHẦN ĐIỀU KHIỂN CỐ ĐỊNH 6 BƯỚC ĐẦU ---
        
        # 5 nước đầu tiên: đi PHẢI (RIGHT)
        if 1 <= step_number <= 5:
            move = Move.RIGHT
            # Kiểm tra nếu vướng tường thì vẫn phải đứng yên hoặc chọn nước khác để tránh crash
            if self._is_traversable(self._apply_move(my_position, move), map_state):
                self.last_pos = my_position
                return move
            
        # Nước thứ 6: đi LÊN (UP)
        if step_number == 6:
            move = Move.UP
            if self._is_traversable(self._apply_move(my_position, move), map_state):
                self.last_pos = my_position
                return move

        # --- TỪ BƯỚC 7 TRỞ ĐI: SỬ DỤNG THUẬT TOÁN ---

        # 1. Cập nhật vị trí kẻ địch (chỉ khi nhìn thấy trong tia 5 ô) [cite: 12, 27]
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # 2. Lấy danh sách nước đi hợp lệ (Chỉ đi vào ô 0, không đi vào ô 1 hay -1) [cite: 31, 35]
        possible_moves = [Move.RIGHT, Move.UP, Move.LEFT, Move.DOWN]
        valid_moves = []
        
        for move in possible_moves:
            next_pos = self._apply_move(my_position, move)
            # QUY TẮC: Không đi vào tường (1), không đi vào ô chưa biết (-1) [cite: 31, 35]
            # Và QUAN TRỌNG: Không đi ngược lại vị trí vừa đứng (tránh loop 2 ô)
            if self._is_traversable(next_pos, map_state) and next_pos != self.last_pos:
                valid_moves.append((move, next_pos))

        # Nếu bị dồn vào đường cùng (chỉ còn cách quay lại hoặc đứng yên)
        if not valid_moves:
            for move in possible_moves:
                next_pos = self._apply_move(my_position, move)
                if self._is_traversable(next_pos, map_state):
                    valid_moves.append((move, next_pos))
            if not valid_moves: return Move.STAY

        # 3. Tính điểm dựa trên khoảng cách tới kẻ địch (nếu có trong bộ nhớ)
        best_move = Move.STAY
        if self.last_known_enemy_pos:
            # BFS chỉ chạy trên các ô đã biết là 0 (đường trống) 
            danger_map = self._compute_bfs_distances(self.last_known_enemy_pos, map_state)
            
            max_dist = -1
            best_candidates = []
            
            for move, n_pos in valid_moves:
                dist = danger_map.get(n_pos, 999) # Nếu không có đường tới, coi như rất xa
                if dist > max_dist:
                    max_dist = dist
                    best_candidates = [move]
                elif dist == max_dist:
                    best_candidates.append(move)
            
            best_move = random.choice(best_candidates)
        else:
            # Nếu chưa từng thấy Pacman, đi tìm ô có tầm nhìn rộng nhất để thám hiểm
            best_move = self._get_exploration_move(my_position, map_state, valid_moves)

        self.last_pos = my_position # Lưu vị trí hiện tại để bước sau không quay lại
        return best_move

    def _is_traversable(self, pos, map_state):
        """Kiểm tra ô có chắc chắn đi được không (phải là ô 0)."""
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _compute_bfs_distances(self, start_pos, map_state):
        """BFS chỉ tính toán trên những ô mà Agent 'biết' là trống (giá trị 0)."""
        distances = {start_pos: 0}
        queue = deque([start_pos])
        while queue:
            curr = queue.popleft()
            d = distances[curr]
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._apply_move(curr, move)
                if self._is_traversable(nxt, map_state) and nxt not in distances:
                    distances[nxt] = d + 1
                    queue.append(nxt)
        return distances

    def _get_exploration_move(self, pos, map_state, valid_moves):
        """Khi không thấy kẻ địch, chọn hướng đi giúp mở rộng vùng nhìn thấy (nhiều ô -1 nhất xung quanh)."""
        best_m = valid_moves[0][0]
        max_unknown = -1
        for move, n_pos in valid_moves:
            # Đếm số ô -1 xung quanh vị trí mới để ưu tiên khám phá [cite: 33]
            unknown_count = 0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r, c = n_pos[0]+dr, n_pos[1]+dc
                    if 0 <= r < 21 and 0 <= c < 21 and map_state[r, c] == -1:
                        unknown_count += 1
            if unknown_count > max_unknown:
                max_unknown = unknown_count
                best_m = move
        return best_m

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)