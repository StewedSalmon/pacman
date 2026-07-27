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
from collections import deque

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
# performance benchmark
import time

class MemoryMap:
    def __init__(self):
        self.known_walls = set()       # các ô chắc chắn là tường
        self.known_empty = set()       # các ô chắc chắn là đường đi
        self.last_seen_enemy = None    # vị trí địch lần cuối thấy
        self.step_last_seen = 0        # bước nào thấy lần cuối
        self.unknown_frontier = set()  # biên giới chưa khám phá

    def update(self, map_state, my_pos, enemy_pos, step_number):
        # Cập nhật bản đồ đã biết
        rows, cols = map_state.shape
        newly_found = set()
        
        # Only check 5x5 area around my_pos
        if map_state[my_pos[0], my_pos[1]] == 0:
            self.known_empty.add(my_pos)
        
        for r in range(max(0, my_pos[0]-5), min(rows, my_pos[0]+6)):
            for c in range(max(0, my_pos[1]-5), min(cols, my_pos[1]+6)):
                if map_state[r][c] == 1:
                    self.known_walls.add((r, c))
                    self.unknown_frontier.discard((r, c))
                elif map_state[r][c] == 0:
                    if (r, c) not in self.known_empty:
                        newly_found.add((r, c))
                    self.known_empty.add((r, c))
                    self.unknown_frontier.discard((r, c))

        # Cập nhật vị trí địch
        if enemy_pos is not None:
            self.last_seen_enemy = enemy_pos
            self.step_last_seen = step_number
            
        # Cập nhật biên giới khám phá (neighbor của known_empty là unknown)
        # self.unknown_frontier.clear()
        for pos in newly_found:
            x, y = pos
            for (dx, dy) in [(-1,0), (1,0), (0,-1), (0,1)]:
                nx, ny = x+dx, y+dy
                if 0 <= nx < rows and 0 <= ny < cols:
                    if map_state[nx][ny] == -1:  # ô chưa biết
                        self.unknown_frontier.add((nx, ny))
            
    def get_safe_neighbors(self, pos, map_state):
        x, y = pos
        rows, cols = map_state.shape
        neighbors = []
        for (dx, dy), move in [
            ((-1,0), Move.UP), ((1,0), Move.DOWN),
            ((0,-1), Move.LEFT), ((0,1), Move.RIGHT)
        ]:
            nx, ny = x+dx, y+dy
            if 0 <= nx < rows and 0 <= ny < cols:
                if map_state[nx][ny] == 0:   # chỉ đi vào ô đã biết là trống
                    neighbors.append(((nx, ny), move))
        return neighbors
    
    def get_exploration_target(self, my_pos, map_state):
        """Tìm mục tiêu khám phá gần nhất trong unknown_frontier."""
        if not self.unknown_frontier:
            # Nếu không có frontier, dùng BFS để tìm ô trống bất kỳ
            return my_pos
        
        # Tìm ô unknown gần nhất bằng Manhattan distance
        best_target = min(
            self.unknown_frontier, 
            key=lambda p: abs(p[0] - my_pos[0]) + abs(p[1] - my_pos[1])
        )
        
        return best_target
    
class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.memory = MemoryMap()
        self.name = "BFS + Memory Pacman"
        self.current_path = []
        self.prev_move = None
        self.visited = set()
        self.last_positions = []

    def step(self, map_state, my_position, enemy_position, step_number):
        start_time = time.perf_counter()

        self.memory.update(map_state, my_position, enemy_position, step_number)
        self.visited.add(my_position)

        # ---- LOOP DETECTION ----
        self.last_positions.append(my_position)
        if len(self.last_positions) > 6:
            self.last_positions.pop(0)

        is_looping = len(set(self.last_positions)) <= 2

        # ---- TARGET ----
        if enemy_position is not None:
            target = enemy_position
        elif self.memory.last_seen_enemy is not None:
            target = self.memory.last_seen_enemy
        else:
            target = self.memory.get_exploration_target(my_position, map_state)

        # ---- REPLAN ----
        if not self.current_path or is_looping:
            path = self._bfs(my_position, target, map_state)

            if not path:
                if target == self.memory.last_seen_enemy:
                    frontier_candidates = list(self.memory.unknown_frontier)
                    if frontier_candidates:
                        frontier_target = min(
                            frontier_candidates,
                            key=lambda p: abs(p[0] - target[0]) + abs(p[1] - target[1])
                        )
                    else:
                        frontier_target = self._bfs_to_frontier(my_position, map_state)
                else:
                    frontier_target = self._bfs_to_frontier(my_position, map_state)

                path = self._bfs(my_position, frontier_target, map_state)

            self.current_path = path

        # ---- FALLBACK ----
        if not self.current_path:
            result = self._explore(my_position, map_state, True)
            # print(f"[Pacman] Step {step_number} | FALLBACK | Time: {time.perf_counter() - start_time:.4f}s")
            return result

        move = self.current_path.pop(0)

        steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
        if steps == 0:
            result = self._explore(my_position, map_state, True)
            # print(f"[Pacman] Step {step_number} | BLOCKED | Time: {time.perf_counter() - start_time:.4f}s")
            return result

        self.prev_move = move

        # multi step
        if self.current_path and self.current_path[0] == move:
            next_pos = self._apply_move(my_position, move)
            next2 = self._apply_move(next_pos, move)

            if self._is_valid_position(next_pos, map_state) and \
            self._is_valid_position(next2, map_state):
                steps = min(2, self.pacman_speed)
                self.current_path.pop(0)

        self.prev_move = move

        result = (move, max(1, steps))
        # print(f"[Pacman] Step {step_number} | target={target} | move={move.name} steps={steps} | Time: {time.perf_counter() - start_time:.4f}s")
        return result
    
    def _bfs(self, start, goal, map_state):
        if start == goal:
            return []
        
        queue = deque([start])
        visited = {start}
        parent = {}

        while queue:
            current = queue.popleft()
            neighbors = self.memory.get_safe_neighbors(current, map_state)

            # tie-break: ưu tiên đi thẳng
            if current in parent:
                _, prev_move = parent[current]
                neighbors.sort(key=lambda x: x[1] != prev_move)

            for next_pos, move in neighbors:
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)

                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)

                    queue.append(next_pos)

        return []

    def _bfs_to_frontier(self, start, map_state):
        queue = deque([start])
        visited = {start}

        while queue:
            current = queue.popleft()

            # nếu gần unknown → target ngon
            if current in self.memory.known_empty:
                for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]:
                    nx, ny = current[0]+dx, current[1]+dy
                    if (nx, ny) in self.memory.unknown_frontier:
                        return current

            for next_pos, _ in self.memory.get_safe_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(next_pos)

        return start  # fallback
    
    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal
        while cur != start:
            prev, move = parent[cur]
            path.append(move)
            cur = prev
        return path[::-1]

    def _explore(self, my_position, map_state, force_random=False):

        neighbors = self.memory.get_safe_neighbors(my_position, map_state)

        if not neighbors:
            return (Move.STAY, 1)

        import random

        if force_random:
            move = random.choice(neighbors)[1]
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            return (move, max(1, steps))

        best_move = Move.STAY
        best_score = -float('inf')

        opposite = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT
        }

        for next_pos, move in neighbors:

            reverse_penalty = -15 if self.prev_move and move == opposite[self.prev_move] else 0
            visited_penalty = -5 if next_pos in self.visited else 3

            # 🔥 gần frontier thì ưu tiên mạnh
            frontier_bonus = 5 if next_pos in self.memory.unknown_frontier else 0

            score = frontier_bonus + visited_penalty + reverse_penalty

            if score > best_score:
                best_score = score
                best_move = move

        steps = self._max_valid_steps(my_position, best_move, map_state, self.pacman_speed)
        self.prev_move = best_move

        return (best_move, max(1, steps))
    
    # Helper methods
    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] == 0

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos

        for _ in range(max_steps):
            dr, dc = move.value
            next_pos = (current[0] + dr, current[1] + dc)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps


class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Strategic Relocation Edition
    Features: Memory Integration, Global Fog Targeting, Minimax Evasion, Zero Ping-Pong.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Grandmaster Ghost"
        self.pacman_speed = kwargs.get('pacman_speed', 2)
        
        if not hasattr(self, 'memory'):
            self.memory = MemoryMap() 
            
        self.tt = {}
        self.map_hash = None
        self.dead_ends = {}
        self.intersections = set()
        
        # MỤC TIÊU KHÁM PHÁ TOÀN CỤC
        self.exploration_target = None

    def step(self, map_state, my_position, enemy_position, step_number):
        self.start_time = time.perf_counter()
        self.TIME_LIMIT = 0.82 
        if len(self.tt) > 50000:
            self.tt.clear()

        # 1. CẬP NHẬT BỘ NHỚ
        self.memory.update(map_state, my_position, enemy_position, step_number)
        threat = enemy_position or self.memory.last_seen_enemy

        # 2. PHÂN TÍCH ĐỊA HÌNH
        wall_mask = (map_state == 1).tobytes()
        current_map_hash = hash(wall_mask)
        if self.map_hash != current_map_hash:
            self._analyze_topology(map_state)
            self.map_hash = current_map_hash

        valid_moves = self._get_neighbors(my_position, map_state)
        
        # Emergency Fallback Guarantee
        if not valid_moves:
            return Move.STAY

        p_root_dist = self._bfs_full(threat, map_state) if threat else {}
        g_root_dist = self._bfs_full(my_position, map_state)
        
        # Kiểm tra khoảng cách nguy hiểm
        danger_dist = p_root_dist.get(my_position, 9999)

        # 3. QUYẾT ĐỊNH CHẾ ĐỘ HOẠT ĐỘNG (Relocate vs Evade)
        # NẾU an toàn (> 8 bước) HOẶC chưa từng thấy Pacman -> Chế độ Chủ động tẩu thoát vào sương mù
        if threat is None or danger_dist > 8:
            
            # Reset target nếu đã đến nơi
            if self.exploration_target == my_position:
                self.exploration_target = None
                
            # Tìm một target sương mù toàn cục mới
            if not self.exploration_target:
                self.exploration_target = self._find_best_fog_target(my_position, threat, map_state, g_root_dist, p_root_dist)
                
            # Đi theo BFS đến target (Kiên định, không ping-pong)
            if self.exploration_target:
                path = self.bfs(my_position, self.exploration_target, map_state)
                if path and path[0] != Move.STAY:
                    total_time = time.perf_counter() - self.start_time
                    # print(f"[Ghost] Step {step_number} | RELOCATING to Fog {self.exploration_target} | Time: {total_time:.4f}s")
                    return path[0]

            # Nếu không tìm được đường tới target mà threat vẫn = None
            # Ta KHÔNG được chạy Minimax (vì không có vị trí Pacman). Thay vào đó, chọn hướng an toàn nhất.
            if threat is None:
                best_fallback = Move.STAY
                best_deg = -1
                for nxt, move in valid_moves:
                    deg = len(self._get_neighbors(nxt, map_state))
                    if deg > best_deg:
                        best_deg = deg
                        best_fallback = move
                total_time = time.perf_counter() - self.start_time
                # print(f"[Ghost] Step {step_number} | FALLBACK (no threat) | Time: {total_time:.4f}s")
                return best_fallback

        # 4. KÍCH HOẠT MINIMAX (Khi nguy hiểm <= 8 bước)
        # Hủy mục tiêu khám phá vì phải lo giữ mạng
        self.exploration_target = None 
        
        best_move = valid_moves[0][1]
        depth = 1
        while True:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break

            move, score = self._search_root(my_position, threat, depth, map_state, p_root_dist, g_root_dist)
            
            if time.perf_counter() - self.start_time <= self.TIME_LIMIT:
                best_move = move
                if score >= 900000 or score <= -900000:
                    break 

            depth += 1

        total_time = time.perf_counter() - self.start_time
        # print(f"[Ghost] Step {step_number} | EVADING Minimax Depth {depth-1} | Time: {total_time:.4f}s")
        return best_move

    def _find_best_fog_target(self, my_pos, threat_pos, map_state, g_dist, p_dist):
        """Tìm 1 ô sương mù AN TOÀN và XA PACMAN nhất trên toàn bản đồ làm đích đến."""
        h, w = map_state.shape
        best_target = None
        best_score = -float('inf')

        for r in range(h):
            for c in range(w):
                if map_state[r, c] == 1: continue # Bỏ qua tường
                if (r, c) in self.dead_ends: continue # Tuyệt đối không chọn đích là ngõ cụt

                # Định nghĩa Sương mù: Những ô chưa từng lưu vào known_empty của bộ nhớ
                is_fog = (r, c) not in self.memory.known_empty and (r, c) not in self.memory.known_walls

                if is_fog:
                    dist_from_me = g_dist.get((r, c), 9999)
                    if dist_from_me < 9999: # Đảm bảo đường đi tới đó không bị kẹt
                        # Tính điểm: Càng xa Pacman càng tốt, nhưng ưu tiên điểm gần Ghost để dễ tới
                        dist_from_threat = p_dist.get((r, c), 0) if threat_pos else 0
                        score = (dist_from_threat * 10) - dist_from_me
                        
                        if score > best_score:
                            best_score = score
                            best_target = (r, c)
                            
        # Nếu đã mở sáng toàn bộ bản đồ (hết sương mù), chọn điểm xa Pacman nhất làm target
        if not best_target and threat_pos:
            for r in range(h):
                for c in range(w):
                    if map_state[r, c] == 1 or (r, c) in self.dead_ends: continue
                    dist_from_me = g_dist.get((r, c), 9999)
                    if dist_from_me < 9999:
                        score = p_dist.get((r, c), 0) * 10 - dist_from_me
                        if score > best_score:
                            best_score = score
                            best_target = (r, c)

        return best_target

    def _search_root(self, ghost_pos, pacman_pos, depth, map_state, p_root_dist, g_root_dist):
        alpha = -float('inf')
        beta = float('inf')
        best_move = Move.STAY
        best_score = -float('inf')
        
        moves = self._get_neighbors(ghost_pos, map_state)
        if not moves:
            return Move.STAY, -1000000 - depth
            
        # Root Move Ordering: Ghost wants to MAXIMIZE distance from Pacman's root
        moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))

        for next_pos, move in moves:
            if time.perf_counter() - self.start_time > self.TIME_LIMIT:
                break
                
            score = self._minimax(next_pos, pacman_pos, depth - 1, False, alpha, beta, map_state, p_root_dist, g_root_dist)
            
            if score > best_score:
                best_score = score
                best_move = move
            alpha = max(alpha, best_score)

        return best_move, best_score

    def _minimax(self, ghost_pos, pacman_pos, depth, is_ghost, alpha, beta, map_state, p_root_dist, g_root_dist):
        # Terminal State: Ghost Caught
        if ghost_pos == pacman_pos:
            return -1000000 - depth  

        # Horizon Reached or Time Out Triggered
        if depth == 0 or time.perf_counter() - self.start_time > self.TIME_LIMIT:
            return self._evaluate(ghost_pos, pacman_pos, p_root_dist, g_root_dist, map_state)

        # Transposition Table Lookup
        tt_key = (ghost_pos, pacman_pos, is_ghost, depth)
        if tt_key in self.tt:
            entry = self.tt[tt_key]
            if entry['type'] == 'exact': return entry['val']
            if entry['type'] == 'lower': alpha = max(alpha, entry['val'])  
            if entry['type'] == 'upper': beta = min(beta, entry['val'])    
            if alpha >= beta: return entry['val']

        orig_alpha = alpha

        if is_ghost:
            best_val = -float('inf')
            moves = self._get_neighbors(ghost_pos, map_state)
            if not moves: 
                return -1000000 - depth
                
            # Move Ordering: Ghost maximizes distance from Pacman's origin
            moves.sort(key=lambda x: -p_root_dist.get(x[0], 0))
            
            for next_pos, _ in moves:
                val = self._minimax(next_pos, pacman_pos, depth - 1, False, alpha, beta, map_state, p_root_dist, g_root_dist)
                best_val = max(best_val, val)
                alpha = max(alpha, best_val)
                if beta <= alpha: break
        else:
            best_val = float('inf')
            moves = self._get_pacman_next_positions(pacman_pos, map_state)
            
            moves.sort(key=lambda x: g_root_dist.get(x, 9999))
            
            for next_pos in moves:
                val = self._minimax(ghost_pos, next_pos, depth - 1, True, alpha, beta, map_state, p_root_dist, g_root_dist)
                best_val = min(best_val, val)
                beta = min(beta, best_val)
                if beta <= alpha: break

        # Transposition Table Store
        tt_type = 'exact'
        if best_val <= orig_alpha: tt_type = 'upper'
        elif best_val >= beta: tt_type = 'lower'
        self.tt[tt_key] = {'val': best_val, 'type': tt_type}
        
        return best_val

    def _evaluate(self, ghost_pos, pacman_pos, p_root_dist, g_root_dist, map_state):
        manhattan = self._manhattan_distance(ghost_pos, pacman_pos)
        
        if manhattan <= 1:
            return -500000

        maze_dist = p_root_dist.get(ghost_pos, 9999)
        score = (maze_dist * 100) + (manhattan * 10)
        
        trap_depth = self.dead_ends.get(ghost_pos, 0)
        if trap_depth > 0:
            score -= (50000 - trap_depth * 100)

        if ghost_pos in self.intersections:
            score += 250  

        if manhattan < 8 and (ghost_pos[0] == pacman_pos[0] or ghost_pos[1] == pacman_pos[1]):
            score -= 3000 

        return score

    def _analyze_topology(self, map_state):
        """
        O(V) single-pass map analysis.
        Identifies dead-ends, their depths, and vital intersections.
        """
        h, w = map_state.shape
        degrees = {}
        self.intersections.clear()
        self.dead_ends.clear()

        # Map degrees and intersections
        for r in range(h):
            for c in range(w):
                if map_state[r, c] != 1:
                    deg = len(self._get_neighbors((r, c), map_state))
                    degrees[(r, c)] = deg
                    if deg >= 3:
                        self.intersections.add((r, c))
        
        # Propagate dead-ends
        dead_ends_init = {pos: 1 for pos, deg in degrees.items() if deg <= 1}
        queue = deque(dead_ends_init.keys())
        self.dead_ends = dead_ends_init.copy()
        
        while queue:
            curr = queue.popleft()
            for nxt, _ in self._get_neighbors(curr, map_state):
                if degrees.get(nxt, 0) == 2 and nxt not in self.dead_ends:
                    self.dead_ends[nxt] = self.dead_ends[curr] + 1
                    queue.append(nxt)

    def _get_pacman_next_positions(self, pacman_pos, map_state):
        positions = set()
        
        for nxt, move in self._get_neighbors(pacman_pos, map_state):
            positions.add(nxt)
            
            if self.pacman_speed > 1:
                dr, dc = move.value
                current = nxt
                for _ in range(1, self.pacman_speed):
                    candidate = (current[0] + dr, current[1] + dc)
                    if self._is_valid_position(candidate, map_state):
                        positions.add(candidate)
                        current = candidate
                    else:
                        break 
                        
        return list(positions) if positions else [pacman_pos]

    def _bfs_full(self, start, map_state):
        dist = {start: 0}
        queue = deque([start])
        while queue:
            pos = queue.popleft()
            d = dist[pos]
            for nxt, _ in self._get_neighbors(pos, map_state):
                if nxt not in dist:
                    dist[nxt] = d + 1
                    queue.append(nxt)
        return dist

    def _manhattan_distance(self, p1: tuple, p2: tuple) -> int:
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w and map_state[r, c] != 1

    def _get_neighbors(self, pos: tuple, map_state: np.ndarray) -> list:
        x, y = pos
        h, w = map_state.shape
        neighbors = []
        if x > 0 and map_state[x-1][y] != 1: neighbors.append(((x-1, y), Move.UP))
        if x < h-1 and map_state[x+1][y] != 1: neighbors.append(((x+1, y), Move.DOWN))
        if y > 0 and map_state[x][y-1] != 1: neighbors.append(((x, y-1), Move.LEFT))
        if y < w-1 and map_state[x][y+1] != 1: neighbors.append(((x, y+1), Move.RIGHT))
        return neighbors

    def _apply_move(self, pos: tuple, move: Move) -> tuple:
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)
        
    def _reconstruct_path(self, parent, start, goal):
        path = []
        cur = goal
        while cur != start:
            cur, move = parent[cur]
            path.append(move)
        return path[::-1]

    def bfs(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        if start == goal: return []
        queue = deque([start]); visited = {start}; parent = {}
        while queue:
            current = queue.popleft()
            for next_pos, move in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    parent[next_pos] = (current, move)
                    if next_pos == goal:
                        return self._reconstruct_path(parent, start, goal)
                    queue.append(next_pos)
        return [Move.STAY]
    
    