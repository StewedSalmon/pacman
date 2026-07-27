import sys
from pathlib import Path
from collections import deque
from heapq import heappop, heappush
import numpy as np
import random
import heapq
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) - Chiến thuật Bản Đồ Nhiệt (Heatmap) Chống Lặp Tuyệt Đối
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "AStar_Heatmap_Seeker"
        
        self.internal_map = np.full((21, 21), -1)
        self.visited = np.zeros((21, 21)) # Bản đồ nhiệt ghi nhớ số lần đã đi qua
        self.last_enemy_pos = None

    def _is_traversable(self, r, c, allow_fog=True):
        if 0 <= r < 21 and 0 <= c < 21:
            val = self.internal_map[r, c]
            if val == 1: return False
            if val == -1: return allow_fog
            return True
        return False

    def step(self, map_state, my_position, enemy_position, step_number):
        # 1. Cập nhật bản đồ sương mù
        visible = map_state != -1
        self.internal_map[visible] = map_state[visible]
        
        # 2. Cộng điểm phạt cho ô hiện tại (Đi càng nhiều phạt càng nặng)
        self.visited[my_position[0], my_position[1]] += 1
        
        # 3. Theo dõi mục tiêu
        if enemy_position:
            self.last_enemy_pos = enemy_position
            
        if self.last_enemy_pos == my_position and not enemy_position:
            self.last_enemy_pos = None
            
        target = enemy_position or self.last_enemy_pos
        
        if not target:
            target = self._find_closest_fog(my_position)
            if not target: return (Move.STAY, 1)
            
        # 4. Tìm đường với A* có tính năng phạt điểm Heatmap
        path = self._astar_heatmap(my_position, target)
        
        # 5. Fallback chống kẹt (chọn đại 1 ô không phải tường nếu A* lỗi)
        if not path or len(path) < 2:
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                r, c = my_position[0] + m.value[0], my_position[1] + m.value[1]
                if self._is_traversable(r, c, allow_fog=False):
                    return (m, 1)
            return (Move.STAY, 1)
            
        # 6. Tối ưu tốc độ di chuyển
        return self._execute_path(path, my_position)

    def _execute_path(self, path, my_position):
        dr, dc = path[1][0] - my_position[0], path[1][1] - my_position[1]
        first_move = Move.STAY
        for m in Move:
            if m.value == (dr, dc):
                first_move = m
                break
                
        steps = 1
        curr = path[1]
        for i in range(2, min(len(path), self.pacman_speed + 1)):
            nxt = path[i]
            # Nếu vẫn đi thẳng hướng đó và không đâm vào tường
            if nxt[0] - curr[0] == dr and nxt[1] - curr[1] == dc:
                if self.internal_map[nxt[0], nxt[1]] == 1:
                    break
                steps += 1
                curr = nxt
            else:
                break
                
        return (first_move, steps)

    def _find_closest_fog(self, start):
        q = deque([start])
        local_visited = {start}
        while q:
            curr = q.popleft()
            if self.internal_map[curr[0], curr[1]] == -1:
                return curr
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                r, c = curr[0] + m.value[0], curr[1] + m.value[1]
                if self._is_traversable(r, c, allow_fog=True) and (r, c) not in local_visited:
                    local_visited.add((r, c))
                    q.append((r, c))
        return None

    def _astar_heatmap(self, start, goal):
        frontier = [(0, start)]
        came_from = {start: None}
        g_score = {start: 0}
        
        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                path = []
                while current:
                    path.append(current)
                    current = came_from[current]
                return path[::-1]
                
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = (current[0] + m.value[0], current[1] + m.value[1])
                
                if self._is_traversable(nxt[0], nxt[1], allow_fog=True):
                    # CHI PHÍ = 1 bước đi + Phạt sương mù + PHẠT BẢN ĐỒ NHIỆT (rất nặng)
                    fog_penalty = 0.5 if self.internal_map[nxt[0], nxt[1]] == -1 else 0
                    heatmap_penalty = self.visited[nxt[0], nxt[1]] * 3  # Phạt x3 cho mỗi lần đã đi qua!
                    
                    cost = 1 + fog_penalty + heatmap_penalty
                    tentative_g = g_score[current] + cost
                    
                    if nxt not in g_score or tentative_g < g_score[nxt]:
                        g_score[nxt] = tentative_g
                        # Heuristic khoảng cách Manhattan
                        f = tentative_g + abs(nxt[0] - goal[0]) + abs(nxt[1] - goal[1])
                        heapq.heappush(frontier, (f, nxt))
                        came_from[nxt] = current
        return None

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
             
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            return self._random_move(my_position, map_state)

        best_move = Move.STAY
        best_score = -float('inf')
        
        # Đánh giá tất cả các hướng đi có thể
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves) # Shuffle để tránh việc AI bị kẹt trong một pattern lặp lại
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            # Bỏ qua nếu là tường
            if not self._is_valid_position(new_pos, map_state):
                continue
                
            # 1. Đánh giá KHÔNG GIAN (Tránh ngõ cụt)
            # Quét xem đằng sau bước đi này có bao nhiêu ô an toàn (tối đa quét 30 ô để tối ưu hiệu năng)
            safe_space = self._get_reachable_area(new_pos, threat, map_state, max_depth=30)
            
            # 2. Đánh giá KHOẢNG CÁCH THỰC TẾ (Không hoảng sợ vô cớ)
            maze_dist = self._bfs_distance(new_pos, threat, map_state)
            
            # CHẤM ĐIỂM BƯỚC ĐI:
            # Ưu tiên số 1: Không gian rộng (tránh ngõ cụt). Chúng ta nhân hệ số lớn (* 100)
            # Ưu tiên số 2: Cách xa Pacman nhất có thể.
            score = (safe_space * 100) + maze_dist
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

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
    def _bfs_distance(self, start: tuple, target: tuple, map_state: np.ndarray) -> int:
        """
        Khắc phục Điểm yếu 1: Tính khoảng cách thực tế trong mê cung bằng BFS.
        """
        if start == target: return 0
        
        queue = deque([(start, 0)])
        visited = set([start])
        
        while queue:
            current, dist = queue.popleft()
            if current == target:
                return dist
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = current[0] + move.value[0], current[1] + move.value[1]
                nxt = (nr, nc)
                
                if self._is_valid_position(nxt, map_state) and nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))
                    
        return float('inf') # Trả về vô cực nếu không có đường tới (VD: bị kẹt kín)

    def _get_reachable_area(self, start: tuple, threat: tuple, map_state: np.ndarray, max_depth: int = 30) -> int:
        """
        Khắc phục Điểm yếu 2: Dùng vết dầu loang (Flood-fill) để đánh giá không gian an toàn.
        Đếm số ô có thể đi tới mà không bị Pacman (threat) cản đường.
        """
        queue = deque([start])
        visited = set([start, threat]) # Coi vị trí của Pacman như một bức tường (không thể đi qua)
        area = 0
        
        while queue and area < max_depth:
            current = queue.popleft()
            area += 1
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = current[0] + move.value[0], current[1] + move.value[1]
                nxt = (nr, nc)
                
                if self._is_valid_position(nxt, map_state) and nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
                    
        return area