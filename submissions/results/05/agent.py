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
import heapq

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


# class PacmanAgent(BasePacmanAgent):
#     """
#     Pacman (Seeker) Agent - Goal: Catch the Ghost
    
#     Implement your search algorithm to find and catch the ghost.
#     Suggested algorithms: BFS, DFS, A*, Greedy Best-First
#     """
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
#         # TODO: Initialize any data structures you need
#         # Examples:
#         # - self.path = []  # Store planned path
#         # - self.visited = set()  # Track visited positions
#         self.name = "Template Pacman"
#         # Memory for limited observation mode
#         self.last_known_enemy_pos = None
    
#     def step(self, map_state: np.ndarray, 
#              my_position: tuple, 
#              enemy_position: tuple,
#              step_number: int):
#         """
#         Decide the next move.
        
#         Args:
#             map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
#             my_position: Your current (row, col) in absolute coordinates
#             enemy_position: Ghost's (row, col) if visible, None otherwise
#             step_number: Current step number (starts at 1)
            
#         Returns:
#             Move or (Move, steps): Direction to move (optionally with step count)
#         """
#         # TODO: Implement your search algorithm here
        
#         # Update memory if enemy is visible
#         if enemy_position is not None:
#             self.last_known_enemy_pos = enemy_position
        
#         # Use current sighting, fallback to last known, or explore
#         target = enemy_position or self.last_known_enemy_pos
        
#         if target is None:
#             # No information about enemy - explore randomly
#             for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
#                 if self._is_valid_move(my_position, move, map_state):
#                     return (move, 1)
#             return (Move.STAY, 1)
        
#         # Example: Simple greedy approach (replace with your algorithm)
#         row_diff = target[0] - my_position[0]
#         col_diff = target[1] - my_position[1]
        
#         # Try to move towards ghost
#         if abs(row_diff) > abs(col_diff):
#             primary_move = Move.DOWN if row_diff > 0 else Move.UP
#             desired_steps = abs(row_diff)
#         else:
#             primary_move = Move.RIGHT if col_diff > 0 else Move.LEFT
#             desired_steps = abs(col_diff)

#         action = self._choose_action(
#             my_position,
#             [primary_move],
#             map_state,
#             desired_steps
#         )
#         if action:
#             return action

#         # If the primary direction is blocked, try other moves
#         fallback_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
#         action = self._choose_action(my_position, fallback_moves, map_state, self.pacman_speed)
#         if action:
#             return action
        
#         return (Move.STAY, 1)
    
#     # Helper methods (you can add more)
    
#     def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
#         for move in moves:
#             max_steps = min(self.pacman_speed, max(1, desired_steps))
#             steps = self._max_valid_steps(pos, move, map_state, max_steps)
#             if steps > 0:
#                 return (move, steps)
#         return None

#     def _max_valid_steps(self, pos: tuple, move: Move, map_state: np.ndarray, max_steps: int) -> int:
#         steps = 0
#         current = pos
#         for _ in range(max_steps):
#             delta_row, delta_col = move.value
#             next_pos = (current[0] + delta_row, current[1] + delta_col)
#             if not self._is_valid_position(next_pos, map_state):
#                 break
#             steps += 1
#             current = next_pos
#         return steps
    
#     def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
#         """Check if a move from pos is valid for at least one step."""
#         return self._max_valid_steps(pos, move, map_state, 1) == 1
    
#     def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
#         """Check if a position is valid (not a wall and within bounds)."""
#         row, col = pos
#         height, width = map_state.shape
        
#         if row < 0 or row >= height or col < 0 or col >= width:
#             return False
        
#         return map_state[row, col] == 0

class PacmanAgent(BasePacmanAgent):
    """
    The Omniscient Predator - Pacman Agent
    Strategy: Particle Filter (Markov Localization) combined with Speed-aware A*.
    Features:
    - Take advantage of visibility in the fog using probabilistic tracking.
    - Corners the enemy at choke points (junctions).
    - Dashes at high speed through the fog by utilizing the static wall map.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Omniscient Predator"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        # Probability matrix for Ghost's location (Belief State)
        self.belief = None

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        h, w = map_state.shape
        
        # 1. INITIALIZE BELIEF STATE (Focus on Top 40% map - Ghost's default spawn)
        if self.belief is None:
            self.belief = np.zeros((h, w), dtype=float)
            # According to environment rules, Ghost spawns in the top 40% of the map
            top_bound = int(h * 0.4)
            valid_spawn_cells = 0
            for r in range(top_bound):
                for c in range(w):
                    # Wall = 1, Traversable != 1
                    if map_state[r, c] != 1: 
                        self.belief[r, c] = 1.0
                        valid_spawn_cells += 1
            if valid_spawn_cells > 0:
                self.belief /= valid_spawn_cells

        # 2. TIME UPDATE: Simulate Ghost's evasive behavior (Probability Diffusion)
        # Only diffuse when the game has started (step > 1) and Ghost is unseen
        if step_number > 1 and enemy_position is None:
            new_belief = np.zeros((h, w), dtype=float)
            for r in range(h):
                for c in range(w):
                    if self.belief[r, c] > 0:
                        transitions = self._simulate_ghost_transition_probs((r, c), my_position, map_state)
                        for (nr, nc), prob in transitions:
                            new_belief[nr, nc] += self.belief[r, c] * prob
            self.belief = new_belief

        # 3. OBSERVATION UPDATE: Update beliefs based on cross-shaped radar
        if enemy_position is not None:
            # Ghost spotted: Assign 100% probability to that exact cell
            self.belief.fill(0.0)
            self.belief[enemy_position[0], enemy_position[1]] = 1.0
            target = enemy_position
        else:
            # Ghost not seen: Eliminate probabilities in currently visible cells (map_state == 0)
            self.belief[map_state == 0] = 0.0
            
            # Normalize the belief matrix
            total_prob = self.belief.sum()
            if total_prob > 0:
                self.belief /= total_prob
            else:
                # Failsafe fallback if tracking is completely lost
                self.belief = (map_state != 1).astype(float)
                self.belief /= self.belief.sum()

            # 4. TARGET SELECTION: Prioritize Choke Points (Junctions) with max probability
            max_prob = np.max(self.belief)
            candidates = np.argwhere(self.belief == max_prob)
            
            def score_candidate(c):
                r, col = c[0], c[1]
                dist = abs(r - my_position[0]) + abs(col - my_position[1])
                
                # Count available exits
                exits = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, col + dc
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                        exits += 1
                
                # Lower score is better. Subtract points for junctions to prioritize them.
                return dist - (exits * 3)
                
            best_target = min(candidates, key=score_candidate)
            target = tuple(best_target)

        # 5. KILL SHOT (Straight-line Sprint)
        # If the enemy is in direct line of sight with no obstacles, dash at max speed
        if enemy_position is not None:
            dist = abs(my_position[0] - enemy_position[0]) + abs(my_position[1] - enemy_position[1])
            direct_mv = self._get_line_of_sight_move(my_position, enemy_position, map_state)
            if direct_mv and dist <= self.pacman_speed:
                return (direct_mv, dist)

        # 6. A* SPEED ENGINE: Find the fastest path to the Target
        move, steps = self._astar_speed_engine(map_state, my_position, target)
        return (move, steps)

    # ==========================================
    # HELPER FUNCTIONS
    # ==========================================

    def _simulate_ghost_transition_probs(self, g_pos, p_pos, map_state):
        """Simulates Ghost's psychology: Prioritizes moving away from Pacman."""
        h, w = map_state.shape
        valid_moves = []
        
        # Ghost can move in 4 cardinal directions or STAY
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]:
            nr, nc = g_pos[0] + dr, g_pos[1] + dc
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                valid_moves.append((nr, nc))
                
        # Calculate Manhattan distance from valid cells to Pacman
        distances = np.array([abs(nr - p_pos[0]) + abs(nc - p_pos[1]) for nr, nc in valid_moves], dtype=float)
        
        # Convert distances to weights (Squared to amplify the desire to escape)
        weights = distances ** 2 
        
        total_weight = weights.sum()
        if total_weight == 0:
            probs = np.ones(len(valid_moves)) / len(valid_moves)
        else:
            probs = weights / total_weight
            
        return list(zip(valid_moves, probs))

    def _astar_speed_engine(self, map_state, start, goal):
        """
        Speed-aware A* Search.
        Considers 1 turn (dashing from 1 to max_speed cells) as cost = 1.
        Safely traverses the fog (-1) because the static wall map (1) is fully known.
        """
        if start == goal:
            return Move.STAY, 1
            
        h, w = map_state.shape
        # Admissible Heuristic = Distance / Speed
        h_func = lambda p: (abs(p[0] - goal[0]) + abs(p[1] - goal[1])) / self.pacman_speed
        
        frontier = [(h_func(start), 0, start, None, None)]
        visited = {start: 0}

        while frontier:
            _, cost, curr, f_move, f_steps = heapq.heappop(frontier)
            if curr == goal:
                return f_move, f_steps

            for move_enum in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move_enum.value
                for n in range(1, self.pacman_speed + 1):
                    nxt = (curr[0] + dr * n, curr[1] + dc * n)
                    
                    # Boundary check
                    if not (0 <= nxt[0] < h and 0 <= nxt[1] < w): 
                        break
                    
                    # Strictly prevent dashing through walls (1)
                    if map_state[nxt[0], nxt[1]] == 1: 
                        break 
                    
                    new_cost = cost + 1
                    if new_cost < visited.get(nxt, float('inf')):
                        visited[nxt] = new_cost
                        priority = new_cost + h_func(nxt)
                        next_f_move = f_move if f_move else move_enum
                        next_f_steps = f_steps if f_steps else n
                        heapq.heappush(frontier, (priority, new_cost, nxt, next_f_move, next_f_steps))
                        
        # Random fallback to avoid crashes if no path is found
        return Move.STAY, 1

    def _get_line_of_sight_move(self, start, end, map_state):
        """Checks for a clear line of sight between Pacman and Ghost."""
        dr, dc = end[0] - start[0], end[1] - start[1]
        if dr != 0 and dc != 0: return None # Not on the same axis
        
        move = Move.UP if dr < 0 else Move.DOWN if dr > 0 else Move.LEFT if dc < 0 else Move.RIGHT
        dist = abs(dr) + abs(dc)
        
        # Check for walls along the path
        for s in range(1, dist):
            chk_r = start[0] + move.value[0] * s
            chk_c = start[1] + move.value[1] * s
            if map_state[chk_r, chk_c] == 1: 
                return None
        return move

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # TODO: Initialize any data structures you need
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        # Belief map to track danger levels across the 21x21 grid
        self.belief_map = np.zeros((21, 21))
        # Standard directions for vision and neighbor checks
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        # Path History (To avoid pacing back and forth)
        self.history = []
        self.max_history = 10 
        # When we lose line-of-sight, we use this to change behavior
        self.stealth_timer = 0
    
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
        # TODO: Implement your search algorithm here

        r, c = my_position
        hallway_row = 10 

        # --- THE "FORCE EXIT" OVERRIDE ---
        if r == hallway_row:
            # Ưu tiên lấy vị trí mối đe dọa (từ đầu hàm step của ông)
            threat = enemy_position or self.last_known_enemy_pos

            # 1. TÌM LỐI THOÁT DỌC (LÊN/XUỐNG) TẠI CHỖ NHƯNG PHẢI AN TOÀN
            safe_exits = []
            for move in [Move.UP, Move.DOWN]:
                if self._is_valid_move(my_position, move, map_state):
                    # Nếu thấy Pacman, không được đi về hướng làm khoảng cách gần lại
                    if threat:
                        next_r = r + move.value[0]
                        next_c = c + move.value[1]
                        dist_now = abs(r - threat[0]) + abs(c - threat[1])
                        dist_next = abs(next_r - threat[0]) + abs(next_c - threat[1])
                        if dist_next < dist_now:
                            continue # Bỏ qua, đi hướng này là đâm đầu vào chỗ chết
                    safe_exits.append(move)
            
            # Có lối dọc an toàn thì đi luôn
            if safe_exits:
                self.history.append(my_position)
                return safe_exits[0]

            # 2. NẾU KHÔNG CÓ LỐI DỌC, TÌM CỘT THOÁT HIỂM AN TOÀN NHẤT
            exit_cols = []
            for col_idx in range(map_state.shape[1]):
                # map_state != 1 là đúng, vì tường luôn là 1 (nhìn xuyên sương mù)
                if map_state[r-1, col_idx] != 1 or map_state[r+1, col_idx] != 1:
                    exit_cols.append(col_idx)

            if exit_cols:
                best_col = None
                min_risk_score = float('inf')
                
                for col in exit_cols:
                    # Điểm rủi ro cơ bản là khoảng cách tới cửa
                    score = abs(col - c)
                    
                    # Nếu Pacman đang chặn ở hướng đó, phạt điểm cực nặng để né
                    if threat:
                        if (c <= col <= threat[1]) or (c >= col >= threat[1]):
                            score += 100 
                    
                    if score < min_risk_score:
                        min_risk_score = score
                        best_col = col

                # 3. DI CHUYỂN NGANG (PHẢI CÓ CHECK TƯỜNG CẢN)
                if best_col is not None:
                    if best_col < c and self._is_valid_move(my_position, Move.LEFT, map_state):
                        self.history.append(my_position)
                        return Move.LEFT
                    elif best_col > c and self._is_valid_move(my_position, Move.RIGHT, map_state):
                        self.history.append(my_position)
                        return Move.RIGHT
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        

        # 1. Update memory and the Belief Map (Danger in the fog)
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            # Reset belief: if we see him, we know exactly where danger is
            self.belief_map.fill(0)
            self.belief_map[enemy_position] = 100
        else:
            # Increase 'danger' score in unseen areas (-1) 
            # This implements the "Belief-Map Evacuation" strategy
            self.belief_map[map_state == -1] += 0.5
            if self.last_known_enemy_pos:
                # Slowly fade the old sighting
                self.belief_map[self.last_known_enemy_pos] *= 0.9 

        # 2. Start the Risk-Aversion Algorithm
        best_move = Move.STAY
        min_risk = float('inf')
        
        # We evaluate all 5 possible actions
        possible_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
        
        for move in possible_moves:
            # Skip moves that hit walls
            if not self._is_valid_move(my_position, move, map_state):
                continue
            
            # Calculate coordinates for the potential move
            delta_r, delta_c = move.value
            new_pos = (my_position[0] + delta_r, my_position[1] + delta_c)
            
            # Evaluate the 'Risk Score' for this new position
            risk = self._calculate_risk(new_pos, map_state, enemy_position)

            # Penalize cells we recently stood in to prevent "vibrating" or staying still too long
            if new_pos in self.history:
                risk += 25 

            # Keep track of the safest option
            if risk < min_risk:
                min_risk = risk
                best_move = move

        # At the very end of step(), update the history
            self.history.append(my_position)
            if len(self.history) > self.max_history:
                self.history.pop(0)

        # 3. Final decision: Return the move with the absolute lowest risk
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
    
    def _calculate_risk(self, pos: tuple, map_state: np.ndarray, enemy_pos: tuple) -> float:
        risk = 0.0
        r, c = pos
        threat = enemy_pos or self.last_known_enemy_pos
        
        # Tốc độ và tầm bắt của Pacman (mặc định lấy 2 để an toàn)
        p_speed = getattr(self, 'pacman_speed', 2) 
        cap_dist = 2 # Capture distance threshold

        if threat:
            tr, tc = threat
            # Khoảng cách Manhattan giữa Ghost và Pacman
            dist = abs(r - tr) + abs(c - tc)
            
            # 1. NGƯỠNG TỬ THẦN (DEATH ZONE)
            # Ngưỡng nguy hiểm = Tốc độ + Tầm bắt (Vd: 2 + 2 = 4 ô)
            danger_threshold = p_speed + cap_dist
            if dist <= danger_threshold:
                # Phạt cực nặng bằng hàm mũ để Ghost ưu tiên quay đầu xe ngay lập tức
                risk += 10000 / (dist + 0.1) 
            elif dist <= 8:
                # Cảnh báo sớm khi Pacman ở gần
                risk += 1000 / dist

            # 2. KIỂM TRA TẦM NHÌN THẲNG (LINE OF SIGHT - LOS)
            # Nếu đứng cùng hàng/cột và KHÔNG có tường chắn -> Pacman speed 2 sẽ "vọt" tới rất nhanh
            if r == tr or c == tc:
                if not self._is_wall_between(pos, threat, map_state):
                    risk += 2000 # Tăng rủi ro vì hành lang không có chỗ nấp

        # 3. PHÂN TÍCH ĐỊA HÌNH (TOPOLOGY)
        neighbors = self._count_neighbors(pos, map_state)
        if neighbors >= 3:
            # "Thưởng" cho ngã ba/tư: Đây là nơi Ghost dễ bẻ lái để Pacman mất dấu (Fog of War)
            risk -= 300 
        elif neighbors <= 1:
            # Ngõ cụt: Phạt nặng nhất vì đây là cái bẫy không lối thoát
            risk += 5000 

        # 4. TRÁNH "LẦY" TẠI CHỖ (ANTI-VIBRATION)
        # Nếu ô đã đứng nhiều lần trong quá khứ thì tăng rủi ro để Ghost di chuyển chỗ mới
        if pos in self.history:
            occurrence = self.history.count(pos)
            risk += (occurrence * 150)

        # 5. CHIẾN THUẬT "TÀU NGẦM" (STEALTH MODE)
        # Ưu tiên di chuyển vào vùng mù (-1) để tận dụng Limited Vision của Pacman
        if map_state[r, c] == -1: # Unseen cell [cite: 33]
            risk -= 100 

        return risk

    def _count_neighbors(self, pos, map_state):
        """Đếm số ô trống có thể đi từ vị trí hiện tại[cite: 31, 32]."""
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] != 1:
                count += 1
        return count
    
    def _is_wall_between(self, pos1: tuple, pos2: tuple, map_state: np.ndarray) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2
        
        # If in the same row, check all columns between them
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1, c] == 1:
                    return True
        # If in the same column, check all rows between them
        elif c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r, c1] == 1:
                    return True
                    
        return False

    def _count_neighbors(self, pos, map_state):
        count = 0
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = pos[0]+dr, pos[1]+dc
            if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] != 1:
                count += 1
        return count
