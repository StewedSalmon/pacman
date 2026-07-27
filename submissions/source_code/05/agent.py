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
import random
import collections
import heapq
from collections import deque
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
    =======================================================================
    V_INTERCEPTOR: THE MATH PREDATOR
    - Mathematical Interception: Calculates the exact intersection point 
      based on Pacman (speed 2) vs Ghost (speed 1).
    - Corners the Ghost at the center (Mid) before it scatters into alleys.
    - O(1) Cache Speed + Hard Loop Prevention (1000 points penalty).
    =======================================================================
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Interceptor"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        
        self.belief = None
        self.history = deque(maxlen=6) 
        
        self.apsp = {} 
        self.valid_cells = []
        self.spawn_center = None
        self.is_map_cached = False
        
        self.rush_mode = True 
        self.meeting_point = None  # Interception point

    # =========================================================
    # 1. CACHE COMPUTATION & GHOST SPAWN DETECTION
    # =========================================================
    def _cache_map_and_spawn(self, map_state):
        h, w = map_state.shape
        self.valid_cells = [(r, c) for r in range(h) for c in range(w) if map_state[r, c] != 1]
        
        # Find the center of the Ghost spawn (Assumed to be in the upper-middle area)
        spawns = [(r, c) for r, c in self.valid_cells if r < h * 0.4 and w * 0.3 < c < w * 0.7]
        self.spawn_center = spawns[len(spawns)//2] if spawns else self.valid_cells[0]

        # O(1) All-Pairs Shortest Path (APSP) computation for instant lookups
        for start in self.valid_cells:
            q = deque([(start, 0)])
            visited = {start}
            while q:
                curr, d = q.popleft()
                self.apsp[(start, curr)] = d
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                    nr, nc = curr[0]+dr, curr[1]+dc
                    if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1 and (nr, nc) not in visited:
                        visited.add((nr, nc))
                        q.append(((nr, nc), d+1))
        self.is_map_cached = True

    def _dist(self, p1, p2):
        return self.apsp.get((p1, p2), 9999)

    # =========================================================
    # 2. DESTRUCTION LOOP (MAIN AGENT LOGIC)
    # =========================================================
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        h, w = map_state.shape
        if not self.is_map_cached: self._cache_map_and_spawn(map_state)

        # Cancel Rush mode if the Ghost is spotted or the interception point is reached but empty
        if enemy_position is not None or my_position == self.meeting_point:
            self.rush_mode = False

        # --- STEP 1: PROBABILITY RADAR (BELIEF STATE) ---
        if self.belief is None:
            self.belief = np.zeros((h, w), dtype=float)
            self.belief[self.spawn_center] = 1.0  # 100% chance Ghost spawns here
            self.belief /= self.belief.sum()

        if enemy_position is not None:
            self.belief.fill(0.0)
            self.belief[enemy_position] = 1.0
            target = enemy_position
        else:
            if step_number > 1:
                new_belief = np.zeros((h, w), dtype=float)
                for r in range(h):
                    for c in range(w):
                        if self.belief[r, c] > 0:
                            moves = [(r+dr, c+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (0,0)]
                                     if 0 <= r+dr < h and 0 <= c+dc < w and map_state[r+dr, c+dc] != 1]
                            dists = [self._dist(m, my_position) for m in moves]
                            weights = np.array(dists, dtype=float) ** 2
                            probs = weights / weights.sum() if weights.sum() > 0 else np.ones(len(moves))/len(moves)
                            for m, prob in zip(moves, probs):
                                new_belief[m] += self.belief[r, c] * prob
                self.belief = new_belief

            self.belief[map_state == 0] = 0.0
            self.belief[my_position] = 0.0  # Force clear current position (Ghost cannot be where Pacman is)

            if self.belief.sum() > 0:
                self.belief /= self.belief.sum()
            else:
                # Failsafe: Reset belief state if tracking is completely lost
                self.rush_mode = False 
                self.belief = (map_state != 1).astype(float)
                self.belief[map_state == 0] = 0.0
                self.belief[my_position] = 0.0
                if self.belief.sum() > 0: self.belief /= self.belief.sum()

            # --- TARGET SELECTION LOGIC (CORE INTERCEPTION) ---
            if self.rush_mode:
                # CALCULATE INTERSECTION POINT BASED ON SPEED 2 (Pacman) vs 1 (Ghost)
                # Find the cell closest to the Ghost spawn that Pacman can reach in time
                best_meet = self.spawn_center
                min_ghost_dist = float('inf')
                
                for cell in self.valid_cells:
                    d_pacman = self._dist(my_position, cell)
                    d_ghost = self._dist(self.spawn_center, cell)
                    
                    # If Pacman reaches this point equal to or before the Ghost (Pacman speed x2)
                    if d_pacman <= d_ghost * 2:
                        # Prefer points as close to the Ghost spawn as possible (choke them at the source)
                        if d_ghost < min_ghost_dist:
                            min_ghost_dist = d_ghost
                            best_meet = cell
                            
                self.meeting_point = best_meet
                target = self.meeting_point
            else:
                # Mid-game: Track the highest probability
                target = np.unravel_index(np.argmax(self.belief), self.belief.shape)

        # --- STEP 2: PURSUIT PATHFINDING (NO STAY, NO LOOPS) ---
        valid_moves = []
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = my_position[0] + m.value[0], my_position[1] + m.value[1]
            if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1:
                valid_moves.append(m)

        if not valid_moves: return Move.STAY, 1 

        # Laser sight kill-shot (Direct line of sight sprint)
        if enemy_position:
            dr, dc = enemy_position[0] - my_position[0], enemy_position[1] - my_position[1]
            if dr == 0 or dc == 0:
                direct_mv = Move.UP if dr < 0 else Move.DOWN if dr > 0 else Move.LEFT if dc < 0 else Move.RIGHT
                dist = abs(dr) + abs(dc)
                clear_sight = True
                for s in range(1, dist):
                    if map_state[my_position[0] + direct_mv.value[0]*s, my_position[1] + direct_mv.value[1]*s] == 1:
                        clear_sight = False; break
                if clear_sight and dist <= self.pacman_speed:
                    return direct_mv, dist

        # Core Anti-Loop path selection
        best_move = valid_moves[0]
        min_cost = float('inf')

        for m in valid_moves:
            endpoint, steps_taken, path_cells = self._simulate_sprint(my_position, m, map_state)
            cost = self._dist(endpoint, target)
            
            # DEATH PENALTY FOR LOOPS: 1000 point penalty for stepping on recent path
            for cell in path_cells:
                if cell in self.history:
                    cost += 1000 

            if cost < min_cost:
                min_cost = cost
                best_move = m

        # Failsafe: If trapped in a corner with all paths penalized, clear history and pick a random escape route
        if min_cost >= 1000:
            self.history.clear() 
            self.rush_mode = False
            best_move = random.choice(valid_moves)

        # Append execution path to history
        _, final_steps, final_path = self._simulate_sprint(my_position, best_move, map_state)
        for cell in final_path:
            self.history.append(cell)

        return best_move, final_steps

    def _simulate_sprint(self, pos, move, map_state):
        """Simulates a sprint based on pacman_speed. Returns endpoint, steps, and traversed cells."""
        steps = 0
        curr = pos
        path_cells = []
        for _ in range(self.pacman_speed):
            nr, nc = curr[0] + move.value[0], curr[1] + move.value[1]
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
                steps += 1
                curr = (nr, nc)
                path_cells.append(curr)
            else:
                break
        return curr, max(1, steps), path_cells


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "ChimeraGhost_AnkleBreaker"
        self.learned_map = None
        self.history = collections.deque(maxlen=20)
        self.last_move = Move.STAY

        self.pacman_speed = int(kwargs.get("pacman_speed", 1))
        self.capture_distance = int(kwargs.get("capture_distance", 1))
        
        self.death_zone = self.capture_distance
        self.danger_zone = self.capture_distance + self.pacman_speed
        self.warning_zone = self.danger_zone + 4

        self.N_PARTICLES = 150 
        self.particles = []

    def _update_map(self, map_state):
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]

    def _update_particles(self, enemy_pos, my_pos):
        h, w = self.learned_map.shape
        valid_cells = [(r, c) for r in range(h) for c in range(w) if self.learned_map[r, c] != 1]

        if enemy_pos is not None:
            self.particles = [enemy_pos] * self.N_PARTICLES
            return enemy_pos

        if not self.particles:
            self.particles = random.choices(valid_cells, k=self.N_PARTICLES)

        new_particles = []
        for p in self.particles:
            if random.random() < 0.8:
                best_nxt = p
                current_d = abs(p[0]-my_pos[0]) + abs(p[1]-my_pos[1]) 
                for _ in range(self.pacman_speed):
                    step_best = best_nxt
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nr, nc = best_nxt[0]+dr, best_nxt[1]+dc
                        if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] != 1:
                            d = abs(nr-my_pos[0]) + abs(nc-my_pos[1])
                            if d < current_d:
                                current_d = d
                                step_best = (nr, nc)
                    best_nxt = step_best
                new_particles.append(best_nxt)
            else:
                opts = [(p[0]+dr, p[1]+dc) for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                        if 0 <= p[0]+dr < h and 0 <= p[1]+dc < w and self.learned_map[p[0]+dr, p[1]+dc] != 1]
                new_particles.append(random.choice(opts) if opts else p)

        self.particles = new_particles
        if not self.particles: return my_pos
        return collections.Counter(self.particles).most_common(1)[0][0]

    def _bfs_dist(self, start, target):
        if not target: return 20
        if start == target: return 0
        h, w = self.learned_map.shape
        queue = collections.deque([(start, 0)])
        visited = {start}
        while queue:
            (r, c), d = queue.popleft()
            if (r, c) == target: return d
            if d > 20: break
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r+dr, c+dc
                if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] != 1:
                    if (nr, nc) not in visited:
                        visited.add((nr, nc))
                        queue.append(((nr, nc), d + 1))
        return 20

    def _count_exits(self, pos):
        h, w = self.learned_map.shape
        return sum(1 for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]
                   if 0 <= pos[0]+dr < h and 0 <= pos[1]+dc < w and self.learned_map[pos[0]+dr, pos[1]+dc] != 1)

    def _corridor_exposure(self, pos, move):
        count = 0
        r, c = pos
        h, w = self.learned_map.shape
        for _ in range(4):
            r += move.value[0]
            c += move.value[1]
            if 0 <= r < h and 0 <= c < w and self.learned_map[r, c] != 1:
                count += 1
            else: break
        return count

    def _is_in_los(self, pos1, pos2):
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]: return False
        r1, c1 = pos1
        r2, c2 = pos2
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if self.learned_map[r1, c] == 1: return False
            return True
        else:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if self.learned_map[r, c1] == 1: return False
            return True

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self._update_map(map_state)
        est_pacman = self._update_particles(enemy_position, my_position)
        current_dist_to_pacman = self._bfs_dist(my_position, est_pacman)

        moves_list = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        best_move = Move.STAY
        best_score = float('-inf')

        opposite = Move.STAY
        if self.last_move == Move.UP: opposite = Move.DOWN
        elif self.last_move == Move.DOWN: opposite = Move.UP
        elif self.last_move == Move.LEFT: opposite = Move.RIGHT
        elif self.last_move == Move.RIGHT: opposite = Move.LEFT

        for move in moves_list:
            nr, nc = my_position[0] + move.value[0], my_position[1] + move.value[1]

            if 0 <= nr < 21 and 0 <= nc < 21 and self.learned_map[nr, nc] != 1:
                score = 0
                next_pos = (nr, nc)
                next_dist = self._bfs_dist(next_pos, est_pacman)
                corridor = self._corridor_exposure(next_pos, move)
                exits = self._count_exits(next_pos)
                visits = self.history.count(next_pos)
                if next_dist <= self.death_zone:
                    score -= 999999 
                elif next_dist <= self.danger_zone:
                    score -= 50000  
                else:
                    score += next_dist * 100
                if current_dist_to_pacman <= self.warning_zone:
                    if move != self.last_move and move != opposite:
                        score += 500
                    elif move == self.last_move:
                        score -= 200
                    score -= visits * 5 
                    
                    # -> Cắt tầm nhìn
                    if self._is_in_los(next_pos, est_pacman):
                        score -= 3000
                        
                    # -> Phá gọng kìm
                    if next_dist < current_dist_to_pacman:
                        score -= 5000
                else:
                    # TRẠNG THÁI AN TOÀN: Đi tuần tra bình thường
                    score -= visits * 300 
                    score += exits * 50   
                    score -= corridor * 20 
                if exits == 0:
                    score -= 999999 
                elif exits == 1:
                    score -= 5000   
                if move == opposite:
                    if exits == 1:
                        score -= 100 
                    else:
                        score -= 80000 
                if score > best_score:
                    best_score = score
                    best_move = move

        self.history.append(my_position)
        self.last_move = best_move
        return best_move
