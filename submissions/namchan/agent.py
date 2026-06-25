"""
24127459 -     - 
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
import time 
import heapq #For priority queue
from collections import deque #For BFS 

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Example Greedy Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # Memory for limited observation mode
        self.last_known_enemy_pos = None

    #All possible position for Pacman to avoid wall
    def get_neighbors(self, pos, map_state):
        neighbor = []
        moves = [Move.UP, Move.DOWN, Move.RIGHT, Move.LEFT]

        for move in moves:
            delta_row, delta_col = move.value
            next_pos = (pos[0] + delta_row, 
                        pos[1] + delta_col)
            if(self._is_valid_position(next_pos, map_state)):
                neighbor.append(next_pos)
        return neighbor
    
    def heuristic(self, pos, goal):
        return abs(goal[0] - pos[0]) + abs(goal[1] - pos[1])
    
    #Use A* for find shortest path to ghost with Manhattan heuristic, g = cost to current, h = cost from current to ghost
    def a_star(self, start, goal, map_state):
        heap = []
        heapq.heappush(heap, (self.heuristic(start, goal) , start))
        g_score = {start: 0}
        parent = {}

        while heap:
            current_f, current = heapq.heappop(heap)
            if current_f > g_score[current] + self.heuristic(current, goal):
                continue
            if(current == goal):
                return parent
            neighbors = self.get_neighbors(current, map_state)
            for move in neighbors:
                g = g_score[current] + 1
                h = self.heuristic(move, goal)
                if move not in g_score or g < g_score[move]:
                    parent[move] = current
                    g_score[move] = g
                    heapq.heappush(heap, (g + h, move))
        return parent
        
    #Real distance cost to any tile in the map
    def bfs(self, start, map_state):
        if not self._is_valid_position(start, map_state):
            return []
        
        rows = len(map_state)
        cols = len(map_state[0])
        cost = [[-1] * cols for _ in range(rows)]
        queue = deque()

        cost[start[0]][start[1]] = 0
        queue.append(start)

        while (queue):
            current = queue.popleft()
            for neighbor in self.get_neighbors(current, map_state):
                r, c = neighbor
                if cost[r][c] != -1:
                    continue
                cost[r][c] = cost[current[0]][current[1]] + 1
                queue.append((r, c))
        return cost        
            
    #Predict where ghost maybe move, assume that ghost will try 
    # to run as far away from pacman as possible in ghost speed = 1 by default(Greedy Escape)

    #This maybe change if we want to block ghost better or perhaps ghost has more complex or clever strategies.
    def predict_ghost(self, ghost_pos, pacman_pos, map_state, previous=None):
        best_pos = ghost_pos
        best_score = -1

        for nxt in self.get_neighbors(ghost_pos, map_state):
            if nxt == previous:
                continue
            dist = self.heuristic(nxt, pacman_pos)
            if dist > best_score:
                best_score = dist
                best_pos = nxt
        return best_pos
    
    #Predict the way ghost maybe move by using many predict_ghost
    def predict_path(self, ghost_pos, pacman_pos, map_state, steps=8):
        path = []
        cur = ghost_pos
        prev = None

        for _ in range(steps):
            nxt = self.predict_ghost(cur, pacman_pos, map_state, prev)
            prev = cur
            cur = nxt
            path.append(cur)
        return path

    #Calculate the cost for pacman moves
    def pacman_travel_time(self, path):
        if len(path) <= 1:
            return 0
        
        turns = 0
        i = 1

        while i < len(path):
            direction = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            run = 1
            j = i + 1
            while j < len(path):
                nd = (path[j][0] - path[j-1][0], path[j][1] - path[j-1][1])
                if nd != direction:
                    break
                run += 1
                j += 1
            turns += (run + self.pacman_speed - 1) // self.pacman_speed
            i = j
        return turns

    #find the location to block ghost
    def find_intercept( self, pacman_pos, ghost_path, ghost_speed, map_state):
        for t, cell in enumerate(ghost_path):
            parent = self.a_star(pacman_pos, cell, map_state)
            path = self.reconstruct_path(parent, pacman_pos, cell)
            if not path:
                continue
            pac_time = self.pacman_travel_time(path)
            ghost_time = (t + 1) / ghost_speed
            if pac_time <= ghost_time:
                return cell
        return ghost_path[-1]

    #From goal find the way back to start
    def reconstruct_path(self, parent, start, goal):
        if goal != start and goal not in parent:
            return []
        
        path = []
        current = goal

        while (current != start):
            path.append(current)
            current = parent[current]

        path.append(start)
        path.reverse()
        return path

    #Desire the direction for pacman and how much possible cells that it can moves
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            # No information about enemy - explore randomly
            return self._explore(my_position, map_state)
        
        ghost_path = self.predict_path(target, my_position, map_state, steps=10)
        intercept = self.find_intercept(my_position, ghost_path, 1, map_state)
        parent = self.a_star(my_position, intercept, map_state)
        path = self.reconstruct_path(parent, my_position, intercept)

        if not path:
            return self._explore(my_position, map_state)
        
        if len(path) < 2:
            return (Move.STAY, 1)

        next_pos = path[1]
        steps = 1
        
        # Calculate direction to target (enemy or last known position)
        row_diff = next_pos[0] - my_position[0]
        col_diff = next_pos[1] - my_position[1]

        if row_diff > 0:
            move = Move.DOWN
        elif row_diff < 0:
            move = Move.UP
        elif col_diff > 0:
            move = Move.RIGHT
        else:
            move = Move.LEFT
            
        for i in range(2, len(path)):
            dr = path[i][0] - path[i - 1][0]
            dc = path[i][1] - path[i - 1][1]
            if (dr == row_diff and dc == col_diff):
                steps += 1
            else: break

        steps = min(steps, self.pacman_speed)
        return (move, steps)

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        
        return (Move.STAY, 1)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0

    def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, desired_steps: int) -> int:
        steps = 0
        max_steps = min(self.pacman_speed, max(1, desired_steps))
        current = pos
        for _ in range(max_steps):
            delta_row, delta_col = move.value
            next_pos = (current[0] + delta_row, current[1] + delta_col)
            if not self._is_valid_position(next_pos, map_state):
                break
            steps += 1
            current = next_pos
        return steps

    def _desired_steps(self, move: Move, row_diff: int, col_diff: int) -> int:
        if move in (Move.UP, Move.DOWN):
            return abs(row_diff)
        if move in (Move.LEFT, Move.RIGHT):
            return abs(col_diff)
        return 1


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._history = []
        self._history_ban = 4  
        self._bfs_cache = {}   
        self.last_known_enemy_pos = None 
        
        # Biến để lưu trữ cấu trúc bản đồ
        self.map_analyzed = False
        self.junctions = set()

    def _analyze_map(self, map_state):
        """Quét ma trận 1 lần duy nhất để tìm tất cả các Ngã 3, Ngã 4 (Vùng an toàn)"""
        if self.map_analyzed:
            return
        rows, cols = map_state.shape
        for r in range(rows):
            for c in range(cols):
                if map_state[r, c] in [0, -1]:
                    # Nếu ô này có từ 3 lối đi trở lên -> Đánh dấu là Junction
                    if self._cell_exits((r, c), map_state) >= 3:
                        self.junctions.add((r, c))
        self.map_analyzed = True

    def _is_valid_position(self, pos, map_state):
        r, c = pos
        return 0 <= r < len(map_state) and 0 <= c < len(map_state[0]) and map_state[r][c] in [0, -1]

    def _get_valid_moves(self, pos, map_state):
        moves = []
        for move, (dr, dc) in [(Move.UP, (-1,0)), (Move.DOWN, (1,0)), (Move.LEFT, (0,-1)), (Move.RIGHT, (0,1))]:
            next_pos = (pos[0] + dr, pos[1] + dc)
            if self._is_valid_position(next_pos, map_state):
                moves.append((move, next_pos))
        return moves

    def _cell_exits(self, pos, map_state):
        return len(self._get_valid_moves(pos, map_state))

    def run_bfs(self, start, map_state):
        if start in self._bfs_cache:
            return self._bfs_cache[start]
        dist = {start: 0}
        q = deque([start])
        while q:
            curr = q.popleft()
            for _, next_pos in self._get_valid_moves(curr, map_state):
                if next_pos not in dist:
                    dist[next_pos] = dist[curr] + 1
                    q.append(next_pos)
        self._bfs_cache[start] = dist
        if len(self._bfs_cache) > 30:
            self._bfs_cache.pop(next(iter(self._bfs_cache)))
        return dist

    def is_in_los(self, p1, p2, map_state):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1][c] == 1: return False
            return True
        if c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r][c1] == 1: return False
            return True
        return False

    def get_voronoi_area(self, g_pos, p_dist_map, map_state):
        g_dist = self.run_bfs(g_pos, map_state)
        safe_area = 0
        for pos, g_d in g_dist.items():
            p_d = p_dist_map.get(pos, 0)
            if g_d < (p_d / 1.5):
                safe_area += 1
        return safe_area

    def find_best_escape_path(self, start_pos, p_dist_map, map_state, enemy_pos):
        start_time = time.time()
        max_time = 0.85 

        global_best_move = Move.STAY
        global_best_eval = (-1, -float('inf')) 
        
        fallback_move = Move.STAY
        max_fallback_dist = -1

        for target_depth in range(2, 11):
            stack = [(start_pos, [], 0, {start_pos})]
            depth_best_move = Move.STAY
            depth_best_eval = (-1, -float('inf'))

            while stack:
                if time.time() - start_time > max_time:
                    return global_best_move if global_best_move != Move.STAY else fallback_move

                curr_pos, path_moves, score, visited = stack.pop()
                
                at_horizon = len(path_moves) >= target_depth
                exits = self._cell_exits(curr_pos, map_state)
                is_tunnel = (exits <= 2)
                
                if at_horizon:
                    if is_tunnel and len(path_moves) < 15:
                        pass 
                    else:
                        area = self.get_voronoi_area(curr_pos, p_dist_map, map_state)
                        final_score = score + (area * 60)
                        
                        # Thưởng thêm nếu đích đến nằm ngay trên một Ngã tư (Vùng an toàn)
                        if curr_pos in self.junctions:
                            final_score += 300

                        eval_tuple = (len(path_moves), final_score)
                        if eval_tuple > depth_best_eval:
                            depth_best_eval = eval_tuple
                            depth_best_move = path_moves[0] if len(path_moves) > 0 else Move.STAY
                        continue

                valid_next = self._get_valid_moves(curr_pos, map_state)
                is_dead_end = True

                for move, next_pos in valid_next:
                    if next_pos not in visited:
                        dist_to_p = p_dist_map.get(next_pos, 0)
                        
                        if len(path_moves) == 0 and dist_to_p > max_fallback_dist:
                            max_fallback_dist = dist_to_p
                            fallback_move = move

                        in_line_of_sight = self.is_in_los(next_pos, enemy_pos, map_state)
                        assumed_pacman_speed = 2.0 if in_line_of_sight else 1.5

                        p_future_dist = dist_to_p - ((len(path_moves) + 1) * assumed_pacman_speed)
                        
                        # FIX BUG: Khoảng cách <= 2.0 là chết, không phải 1.0!
                        if p_future_dist <= 2.0:
                            eval_tuple = (len(path_moves), score)
                            if eval_tuple > depth_best_eval:
                                depth_best_eval = eval_tuple
                                depth_best_move = path_moves[0] if len(path_moves) > 0 else move
                            continue 

                        is_dead_end = False
                        step_score = dist_to_p
                        
                        # Lực hút của Ngã tư: Rút ngắn khoảng cách tới các ngã tư
                        if next_pos in self.junctions:
                            step_score += 150
                            
                        # Thả thính Pacman nếu an toàn
                        if 4 <= dist_to_p <= 6 and next_pos in self.junctions:
                            step_score += 200  

                        if in_line_of_sight:
                            step_score -= 200 # Phạt Line of Sight nặng hơn để tránh bị rượt thẳng

                        new_visited = set(visited)
                        new_visited.add(next_pos)
                        stack.append((next_pos, path_moves + [move], score + step_score, new_visited))

                if is_dead_end and len(path_moves) > 0:
                    eval_tuple = (len(path_moves), score - 2000)
                    if eval_tuple > depth_best_eval:
                        depth_best_eval = eval_tuple
                        depth_best_move = path_moves[0]

            if depth_best_eval[0] != -1:
                global_best_eval = depth_best_eval
                global_best_move = depth_best_move

        return global_best_move if global_best_move != Move.STAY else fallback_move

    def step(self, map_state, my_position, enemy_position, step_number):
        me = tuple(int(v) for v in my_position)

        # Kích hoạt hàm đọc Map ở lượt đầu tiên
        self._analyze_map(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(int(v) for v in enemy_position)
            
        target_threat = enemy_position or self.last_known_enemy_pos

        if target_threat is None:
            moves = self._get_valid_moves(me, map_state)
            return random.choice(moves)[0] if moves else Move.STAY

        p_pos = tuple(int(v) for v in target_threat)
        p_dist_map = self.run_bfs(p_pos, map_state)

        chosen_move = self.find_best_escape_path(me, p_dist_map, map_state, p_pos)
        
        if chosen_move != Move.STAY:
            for m, n_p in self._get_valid_moves(me, map_state):
                if m == chosen_move:
                    self._history.append(n_p)
                    if len(self._history) > self._history_ban:
                        self._history.pop(0)

        return chosen_move