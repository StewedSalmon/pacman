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

def get_mahattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculates the Manhattan distance between two positions."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

class PacmanAgent(BasePacmanAgent):
    """Markov Localization"""
    def __init__(self, **kwargs):
        """Initialize the Pacman agent.
        Students can set up any data structures they need here."""
        super().__init__(**kwargs)
        self.name = "Minimax Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.max_depth = 3
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.map_BFS_cache = {}

        self.height = kwargs.get("map_height", 21)
        self.width = kwargs.get("map_width", 21)
        self.explored_map = np.full((self.height, self.width), -1)  # -1 for unexplored, 0 for empty, 1 for wall
        self.heatmap = np.zeros((self.height, self.width)) # heatmap for belief state
        self.time_last_seen_enemy = 0 # track how long since we last saw the enemy
        self.is_hunting_blind = False
        
        self.early_game = True
        self.previous_pos = None
        self.patrol_target = None
        
    def get_BFS_map(self, start: tuple):
        """Retrieves or calculates BFS map from start position"""
        if start in self.map_BFS_cache:
            return self.map_BFS_cache[start]
        
        distances = {start: 0}
        parent = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                x, y = move.value
                next = (current[0] + x, current[1] + y)
                if self.is_tile_valid(next) and next not in distances:
                    distances[next] = distances[current] + 1
                    parent[next] = (current, move)
                    queue.append(next)
        self.map_BFS_cache[start] = (distances, parent)
        return self.map_BFS_cache[start] 
    
    def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, pacman_turn: bool):
        # base case
        if ghost_position == pacman_position:
           return 0 - depth
        if depth == 0: 
            distances, parent = self.get_BFS_map(pacman_position)
            if ghost_position not in distances:
                return float('inf')
            
            # reconstruct path
            path = []
            current = ghost_position
            while current != pacman_position:
                previous, move_taken = parent[current]
                path.append(move_taken)
                current = previous
            path.reverse() 
            if len(path) > 0:
                capture_path = path[:-1] # remove the last tile since it is the capture tile
            else:
                capture_path = path
            # calculate ticks   
            ticks = self.pacman_tick(capture_path)
            manhattan_distance = get_mahattan_distance(ghost_position, pacman_position)
            return (ticks * 10) + len(capture_path) + manhattan_distance * 0.1 # use path length and manhattan distance as tie breakers
        
        # minimizing for pacman
        if pacman_turn:
            min_eval = float('inf')
            pacman_next_positions = self.get_sprint_options(pacman_position)         
            for _, _, next_pacman_pos in pacman_next_positions:
                eval = self.minimax(ghost_position, next_pacman_pos, map_state, depth - 1, alpha, beta, False)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
            
        # maximizing for ghost
        else:
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                
                if self.is_tile_valid(next_ghost_pos):
                    eval = self.minimax(next_ghost_pos, pacman_position, map_state, depth - 1, alpha, beta, True)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        
    def explore_map(self, my_position: tuple, map_state: np.ndarray):
        """Run BFS to find the nearest frontier"""
        if self.early_game:
            if my_position[0] <= 9:
                self.early_game = False
            else:
                center_target = (9, 10)
                best_move = Move.STAY
                best_steps = 1
                min_dist = float('inf')
                
                options = self.get_sprint_options(my_position)
                for move, steps, landing_pos in options:
                    if move == Move.STAY:
                        continue
                    
                    row_dist = abs(landing_pos[0] - center_target[0])
                    col_dist = abs(landing_pos[1] - center_target[1])
                    dist = row_dist * 2 + col_dist
                    
                    if hasattr(self, 'previous_pos') and landing_pos == self.previous_pos:
                        dist += 100
                    
                    # Tie-breaker: if distance is equal, prefer the move that takes more steps (sprinting)
                    if dist < min_dist or (dist == min_dist and steps > best_steps):
                        min_dist = dist
                        best_move = move
                        best_steps = steps
                
                # If a valid move towards the center exists, take it. 
                # Otherwise, something is blocking us, so turn off early game.
                if best_move != Move.STAY:
                    return (best_move, best_steps)
                else:
                    self.early_game = False

        # --- NORMAL FRONTIER EXPLORATION ---
        self.get_BFS_map(my_position)
        distance, parent = self.map_BFS_cache.get(my_position, ({}, {}))
        
        frontier_positions = [pos for pos in distance if self.is_frontier(pos)]
        if frontier_positions:
            # Find the nearest frontier
            target_pos = min(frontier_positions, key=lambda pos: distance[pos])
        else:
            if len(distance) < 1:
                return Move.STAY
            # If we don't have a target, or we reached it, pick a new farthest point
            if not hasattr(self, 'patrol_target') or self.patrol_target is None or self.patrol_target == my_position:
                self.patrol_target = max(distance, key=lambda pos: distance[pos])
                
            target_pos = self.patrol_target
        
        # Reconstruct path to the nearest frontier
        path = []
        current = target_pos
        while current != my_position:
            previous, move_taken = parent[current]
            path.append(move_taken)
            current = previous
        path.reverse()
        
        if len(path) > 0:
            next_move = path[0]
            
            # SAFE SPRINTING: Count how many consecutive steps the optimal path takes in this direction
            consecutive_path_steps = 0
            for move in path:
                if move == next_move:
                    consecutive_path_steps += 1
                else:
                    break
            
            # Match the chosen move against sprint options, capping at the consecutive_path_steps
            options = self.get_sprint_options(my_position)
            best_steps = 1
            for move, steps, _ in options:
                if move == next_move and steps <= consecutive_path_steps and steps > best_steps:
                    best_steps = steps
                    
            return (next_move, best_steps)
        else:
            return (Move.STAY, 1)
        
    def get_diffusion_neighbours(self, pos: tuple, include_stay=True) -> list:
        """Get valid neighbouring positions for Ghost diffusion (allows spreading into the fog)."""
        neighbours = []
        if include_stay:
            neighbours.append(pos)
            
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            nr, nc = pos[0] + delta_row, pos[1] + delta_col
            
            if 0 <= nr < self.height and 0 <= nc < self.width:
                # The Ghost can move into both explored space (0) AND unexplored fog (-1)
                # It just can't move into known walls (1).
                if self.explored_map[nr, nc] != 1:
                    neighbours.append((nr, nc))
                    
        return neighbours
        
    def update_heatmap(self, current_visible_cells: set, enemy_pos: tuple):
        """Update the heatmap based on the last known position of the enemy and time since last seen."""
        if enemy_pos is not None: # if enemy position is visible
            self.heatmap.fill(0)  # Reset heatmap
            self.heatmap[enemy_pos] = 100  
            self.is_hunting_blind = False
            return
        
        if not self.is_hunting_blind and self.last_known_enemy_pos is not None: # turn 0: init heat map
            self.is_hunting_blind = True
            self.heatmap.fill(0)
            self.heatmap[self.last_known_enemy_pos] = 100  # Start with the last known position
            
        if not self.is_hunting_blind:
            return
        
        # equal diffusion
        new_heatmap = np.zeros_like(self.heatmap)
        for r in range(self.height):
            for c in range(self.width):
                if self.heatmap[r, c] > 0:
                    # Spread the heat to neighbouring cells
                    valid_neighbours = self.get_diffusion_neighbours((r, c), include_stay=True)
                    if valid_neighbours:
                        spread_value = self.heatmap[r, c] / len(valid_neighbours)
                        for nr, nc in valid_neighbours:
                            new_heatmap[nr, nc] += spread_value
                            
        self.heatmap = new_heatmap
        
        # vision and wall collapse: remove heat from cells that are visible or are walls
        self.heatmap[self.explored_map == 1] = 0
        for r, c in current_visible_cells:
            self.heatmap[r, c] = 0
            
        # normalization
        total_heat = np.sum(self.heatmap)
        if total_heat > 0:
            self.heatmap = (self.heatmap / total_heat) * 100  # Normalize to make it a probability distribution
        else:
            # reset to uniform distribution of all non-wall
            non_walls = (self.explored_map != 1)
            count = np.sum(non_walls)
            if count > 0:
                self.heatmap[non_walls] = 100 / count
    
    def update_explored_map(self, map_state: np.ndarray):
        """Update the explored map based on the current position and map state."""
        for r in range(self.height):
            for c in range(self.width):
                if map_state[r, c] != -1:
                    if map_state[r, c] == 2 or map_state[r, c] == 3: # 2 is Pacman, 3 is Ghost
                        self.explored_map[r, c] = 0  # Mark agent positions as valid
                    else:
                        self.explored_map[r, c] = map_state[r, c]
        
    def is_tile_valid(self, pos: tuple):
        """Check if a position is valid (not a wall)"""
        row, col = pos
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.explored_map[row, col] == 0
        return False
    
    def is_frontier(self, pos: tuple):
        """Check if a position is a frontier (adjacent to unexplored tile)"""
        row, col = pos
        if not self.is_tile_valid(pos):
            return False
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (row + move.value[0], col + move.value[1])
            if 0 <= next_pos[0] < self.height and 0 <= next_pos[1] < self.width:
                if self.explored_map[next_pos[0], next_pos[1]] == -1: 
                    return True
        return False
    
    def get_valid_neighbours(self, pos: tuple, include_stay=True) -> list:
        """Get valid neighbouring positions (not walls)"""
        neighbours = []
        if include_stay:
            neighbours.append(pos)  # Include the current position as a valid neighbour
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            next_pos = (pos[0] + delta_row, pos[1] + delta_col)
            if self.is_tile_valid(next_pos):
                neighbours.append(next_pos)
        return neighbours
    
    def get_visible_cells(self, pos: tuple) -> set:
        """Get all visible cells from the current position (up, down, left, right until a wall)"""
        visible_cells = {pos}
        directions = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        
        for move in directions:
            delta_row, delta_col = move.value
            row, col = pos
            for _ in range(5):
                row += delta_row
                col += delta_col
                if not (0 <= row < self.height and 0 <= col < self.width): # out of bounds
                    break
                if self.explored_map[row, col] == 1: # wall
                    visible_cells.add((row, col))  # Add the wall cell to visible cells
                    break
                if self.explored_map[row, col] == -1: # unexplored
                    break
                visible_cells.add((row, col))
                
        return visible_cells
    
    def evaluate_blind_move(self, pos: tuple):
        """Return best move when the enemy is missing"""
        
        # Run BFS to get accurate TravelCost where possible
        self.get_BFS_map(pos)
        distances, parent = self.map_BFS_cache.get(pos, ({}, {}))
        
        best_target = pos
        max_score = -float('inf')
        
        # one move away costs 2% probability
        LAMBDA = 2.0
        
        # target the region with the higest RegionScore
        for r in range(self.height):
            for c in range(self.width):
                if self.heatmap[r, c] > 0:
                    r_min, r_max = max(0, r-1), min(self.height, r+2)
                    c_min, c_max = max(0, c-1), min(self.width, c+2)
                    
                    prob_mass = np.sum(self.heatmap[r_min:r_max, c_min:c_max])
                    
                    if (r, c) in distances:
                        travel_cost = distances[(r, c)]
                    else:
                        travel_cost = get_mahattan_distance(pos, (r, c))
                        
                    score = prob_mass - (LAMBDA * travel_cost)
                    
                    if score > max_score:
                        max_score = score
                        best_target = (r, c)
        
        if best_target == pos:
            valid_neighbours = self.get_valid_neighbours(pos, include_stay=False)
            if valid_neighbours:
                target = random.choice(valid_neighbours)
                return (Move((target[0] - pos[0], target[1] - pos[1])), 1)
            return (Move.STAY, 1)
        
        # reconstruct path to best target
        if best_target in distances:
            path = []
            current = best_target
            while current != pos:
                previous, move_taken = parent[current]
                path.append(move_taken)
                current = previous
            path.reverse()
            
            if len(path) > 0:
                next_move = path[0]
                
            # Determine how many consecutive steps the path takes in this direction
                consecutive_steps = 0
                for m in path:
                    if m == next_move:
                        consecutive_steps += 1
                    else:
                        break
                        
                # Match against sprint options to safely execute multi-step moves
                options = self.get_sprint_options(pos)
                best_steps = 1
                for move, steps, _ in options:
                    if move == next_move and steps <= consecutive_steps and steps > best_steps:
                        best_steps = steps
                        
                return (next_move, best_steps)
            
        # if no path: pick the move that reduces Manhattan distance the most
        best_move = Move.STAY
        best_steps = 1
        min_dist = float('inf')
            
        options = self.get_sprint_options(pos)
        for move, steps, landing_pos in options:
            if move == Move.STAY:
                continue
            dist = get_mahattan_distance(landing_pos, best_target)
                
            # Tie-breaker: if distance is equal, prefer the move that takes more steps (sprinting)
            if dist < min_dist or (dist == min_dist and steps > best_steps):
                min_dist = dist
                best_move = move
                best_steps = steps
                    
        if best_move != Move.STAY:
            return (best_move, best_steps)
            
        return (Move.STAY, 1)
                
    def pacman_tick(self, path: list) -> int:
        if not path:
            return 0
        ticks = 0
        current_move = path[0]
        steps = 0
        
        for move in path:
            if move == current_move and steps < self.pacman_speed:
                steps += 1
            else:
                ticks += 1
                current_move = move
                steps = 1
        if steps > 0:
            ticks += 1
        return ticks
    
    def get_sprint_options(self, pos: tuple):
        """
        Generates all valid moves and step counts for Pacman.
        Returns a list of tuples: (Move, step_count, landing_position)
        """
        options = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            if move == Move.STAY:
                options.append((move, 1, pos))
                continue
            delta_row, delta_col = move.value
            current_pos = pos        
            for steps in range(1, self.pacman_speed + 1):
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                # Stop projecting if we hit a wall
                if not self.is_tile_valid(next_pos):
                    break 
                current_pos = next_pos
                options.append((move, steps, current_pos))
        return options
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        self.map_BFS_cache.clear()
        
        # Update explored map with current position and map state
        self.update_explored_map(map_state)
        current_vision = self.get_visible_cells(my_position)
        self.update_heatmap(current_vision, enemy_position)
        
        chosen_action = Move.STAY
        
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.time_last_seen_enemy = 0
            
            best_move = Move.STAY
            best_steps = 1
            min_eval = float('inf')
            alpha = -float('inf')
            beta = float('inf')
            
            possible_moves = self.get_sprint_options(my_position)
            for move, steps, landing_pos in possible_moves:
                eval_score = self.minimax(enemy_position, landing_pos, map_state, self.max_depth, alpha, beta, False)
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_move = move
                    best_steps = steps
                beta = min(beta, eval_score)
                
            if best_steps > 1:
                chosen_action = (best_move, best_steps)
            else:
                chosen_action = Move.STAY
            
        else:
            self.time_last_seen_enemy += 1
            if self.last_known_enemy_pos is None or self.time_last_seen_enemy > 5: 
                # Start of game: we have literally no info, expand frontiers
                chosen_action = self.explore_map(my_position, map_state)
            else: 
                # Blind hunting logic using the active heatmap
                chosen_action = self.evaluate_blind_move(my_position)
            
        self.previous_pos = my_position
        return chosen_action

from enum import Enum
class GhostState(Enum):
    OPENING = 1
    TRANSIT = 2
    ARRIVED = 3

class GhostAgent(BaseGhostAgent):
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        """
        super().__init__(**kwargs)
        self.name = "Modular Evasion Ghost"
        # CONFIG
        self.config = {
            'first_move': 'commit',       # 'observe' (wait) or 'commit' (move immediately)
            'observe_duration': 1, # between 1 and 3, only activated if in 'observe' mode
            'target_area': 'closest',   # 'furthest', 'closest'
            'post_arrival': 'wander',       # 'stay' (hide), 'wander' (walk randomly), or 'loop' (back to transit and go to other corners)
            'loop_timing': 'continuous',     # 'continuous' (never stop), 'periodic' (loop after X turns) TODO: implement this
        }
        
        # SAFE ZONES
        self.target_waypoints = {
            'quadrant_1': (1, 1),
            'quadrant_2': (1, 19),
            'quadrant_3': (19, 1),
            'quadrant_4': (19, 19),
            'special': (15, 10) # pacman spawn point
        }
        
        self.explored_waypoints = []
        
        self.pacman_speed = kwargs.get("pacman_speed", 1)
        self.max_depth = 2
        self.map_BFS_cache = {}
        self.dead_ends = set()
        
        self.height = kwargs.get("map_height", 21)
        self.width = kwargs.get("map_width", 21)
        self.explored_map = np.full((self.height, self.width), -1)  
        self.heatmap = np.zeros((self.height, self.width)) 
        
        # State Machine Memory
        self.state = GhostState.OPENING
        self.target_node = None
        
        # Evasion tracking
        self.last_known_enemy_pos = None
        self.time_last_seen_enemy = 0 
        self.is_hunting_blind = False
        
        self.previous_step = None
        
    def get_BFS_map(self, start: tuple, safe_only=True):
        """Retrieves or calculates BFS map from start position"""
        if start in self.map_BFS_cache:
            return self.map_BFS_cache[start]
        
        distances = {start: 0}
        parent = {start: None}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                x, y = move.value
                next_pos = (current[0] + x, current[1] + y)
                is_valid = self.is_tile_safe(next_pos) if safe_only else self.is_tile_valid(next_pos)
                
                if is_valid and next_pos not in distances:
                    distances[next_pos] = distances[current] + 1
                    parent[next_pos] = (current, move)
                    queue.append(next_pos)
        self.map_BFS_cache[start] = (distances, parent)
        return self.map_BFS_cache[start]
    
    def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, ghost_turn: bool):
        # base case
        if get_mahattan_distance(ghost_position, pacman_position) < 2:
            return -1000 - depth
        
        if depth == 0:
            distances, parent = self.get_BFS_map(pacman_position)
            if ghost_position not in distances:
                return float('inf')
            
            # reconstruct path
            path = []
            current = ghost_position
            while current != pacman_position:
                previous, move_taken = parent[current]
                path.append(move_taken)
                current = previous
            path.reverse()
            
            if len(path) > 0:
                capture_path = path[:-1] # remove the last tile since it is the capture tile
            else:
                capture_path = path
            
            # calculate score
            ticks = self.pacman_tick(capture_path)
            score = (ticks * 10) + len(capture_path) # use path length as tiebreaker
            return score
        
        # maximizing for ghost
        if ghost_turn:
            max_eval = -float('inf')
            next_options = []
            for move in [Move.STAY, Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                if self.is_tile_valid(next_ghost_pos):
                    next_options.append(next_ghost_pos)
            next_options.sort(key=lambda pos: abs(pos[0] - pacman_position[0]) + abs(pos[1] - pacman_position[1]), reverse=True)
            
            for next_ghost in next_options:
                eval = self.minimax(next_ghost, pacman_position, map_state, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        
        # minimizing for pacman
        else:
            min_eval = float('inf')
            pacman_next_positions = self.pacman_simulated_positions(pacman_position)
            pacman_next_positions.sort(key=lambda pos: abs(pos[0] - ghost_position[0]) + abs(pos[1] - ghost_position[1]))
            for next_pacman_pos in pacman_next_positions:
                eval = self.minimax(ghost_position, next_pacman_pos, map_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def update_explored_map(self, map_state: np.ndarray):
            """Update the explored map based on the current position and map state."""
            for r in range(self.height):
                for c in range(self.width):
                    if map_state[r, c] != -1:
                        if map_state[r, c] == 2 or map_state[r, c] == 3: # 2 is Pacman, 3 is Ghost
                            self.explored_map[r, c] = 0  # Mark agent positions as valid
                        else:
                            self.explored_map[r, c] = map_state[r, c]
                            
    def update_dead_ends(self):
        """Dynamically identifies and tags single-entry dead ends."""
        changed = True
        while changed:
            changed = False
            for r in range(self.height):
                for c in range(self.width):
                    if self.explored_map[r, c] != 1 and (r, c) not in self.dead_ends:
                        # Count valid exits that are not already tagged as dead ends
                        open_exits = 0
                        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                            nr, nc = r + move.value[0], c + move.value[1]
                            if 0 <= nr < self.height and 0 <= nc < self.width:
                                if self.explored_map[nr, nc] != 1 and (nr, nc) not in self.dead_ends:
                                    open_exits += 1
                                    
                        # If a tile only has 1 valid exit, it is a dead end.
                        if open_exits <= 1:
                            self.dead_ends.add((r, c))
                            changed = True
            
    def is_tile_valid(self, pos: tuple):
        """Check if a position is valid (not a wall)"""
        row, col = pos
        if 0 <= row < self.height and 0 <= col < self.width:
            return self.explored_map[row, col] == 0
        return False
    
    def is_tile_safe(self, pos: tuple):
        return self.is_tile_valid(pos) and pos not in self.dead_ends
    
    def get_visible_cells(self, pos: tuple) -> set:
        """Get all visible cells from the current position in straight lines."""
        visible_cells = {pos}
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            row, col = pos
            for _ in range(self.width): # Look indefinitely until a wall
                row += delta_row
                col += delta_col
                if not (0 <= row < self.height and 0 <= col < self.width):
                    break
                if self.explored_map[row, col] == 1 or self.explored_map[row, col] == -1: # Wall blocks vision
                    break
                visible_cells.add((row, col))
        return visible_cells

    def update_heatmap(self, current_visible_cells: set, enemy_pos: tuple):
        """Diffuses heat based on Pacman's speed and clears visible cells."""
        if enemy_pos is not None:
            self.heatmap.fill(0)
            self.heatmap[enemy_pos] = 100
            self.is_hunting_blind = False
            return
            
        if not self.is_hunting_blind and self.last_known_enemy_pos is not None:
            self.is_hunting_blind = True
            self.heatmap.fill(0)
            self.heatmap[self.last_known_enemy_pos] = 100
            
        if not self.is_hunting_blind:
            return

        # Diffuse heat. Because Pacman has speed 2, we diffuse slightly faster/wider.
        new_heatmap = np.zeros_like(self.heatmap)
        for r in range(self.height):
            for c in range(self.width):
                if self.heatmap[r, c] > 0:
                    # Quick BFS to simulate Pacman's sprint diffusion
                    queue = [(r, c)]
                    visited = {(r, c)}
                    for _ in range(self.pacman_speed): 
                        next_q = []
                        for curr_r, curr_c in queue:
                            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                                nr, nc = curr_r + move.value[0], curr_c + move.value[1]
                                if self.is_tile_valid((nr, nc)) and (nr, nc) not in visited:
                                    visited.add((nr, nc))
                                    next_q.append((nr, nc))
                        queue = next_q
                    
                    spread_value = self.heatmap[r, c] / len(visited)
                    for vr, vc in visited:
                        new_heatmap[vr, vc] += spread_value

        self.heatmap = new_heatmap
        
        # Ghost clears heat from tiles it can currently see
        self.heatmap[self.explored_map == 1] = 0
        for r, c in current_visible_cells:
            self.heatmap[r, c] = 0
            
        # normalize heat map   
        total_heat = np.sum(self.heatmap)
        if total_heat > 0:
            self.heatmap = (self.heatmap / total_heat) * 100
        else:
            # reset to a uniform distribution across all known non-wall tiles
            non_walls = (self.explored_map == 0)
            count = np.sum(non_walls)
            if count > 0:
                self.heatmap[non_walls] = 100 / count

    def evaluate_evasion_move(self, pos: tuple) -> Move:
        """Finds the best hiding spot when Pacman is not visible."""
        best_move = Move.STAY
        best_score = -float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            
            if not self.is_tile_valid(next_pos):
                continue
                
            score = 0
                
            # 1. Evaluate visibility from the target tile
            visible_from_next = self.get_visible_cells(next_pos)
            
            # 2. Heat penalty: How much danger can see this tile?
            heat_in_los = sum([self.heatmap[r, c] for r, c in visible_from_next])
            score -= (heat_in_los * 10) 
            
            # 3. Exposure penalty: Encourage corners/hiding. 
            # Fewer visible cells = better corner.
            exposure = len(visible_from_next)
            score -= exposure 
            
            # 4. Distance bonus: Keep moving generally away from last known
            if self.last_known_enemy_pos:
                dist = get_mahattan_distance(next_pos, self.last_known_enemy_pos)
                score += (dist * 0.5)
                
            # Anti-stalling: Slightly penalize backtracking if we can avoid it
            # (Assuming you track self.last_pos somewhere; if not, ignore this part)
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move
        
    def pacman_tick(self, path: list) -> int:
        if not path:
            return 0
        ticks = 0
        current_move = path[0]
        steps = 0
        
        for move in path:
            if move == current_move and steps < self.pacman_speed:
                steps += 1
            else:
                ticks += 1
                current_move = move
                steps = 1
        if steps > 0:
            ticks += 1
        return ticks
        
    def pacman_simulated_positions(self, pacman_position: tuple):      
        """Generate sensible next moves for Pacman based on current position and map state."""
        simulated_positions = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
            row, col = move.value
            current_position = pacman_position
            
            for step in range(self.pacman_speed):
                next_position = (current_position[0] + row, current_position[1] + col)
                
                if not self.is_tile_valid(next_position):
                    break
                current_position = next_position
                
                if current_position != pacman_position:
                    simulated_positions.append(current_position)
        return simulated_positions
    
    def pick_target_area(self, pos: tuple):
        """Select a destination based on target area config"""
        valid_targets = [coord for name, coord in self.target_waypoints.items() 
                         if coord != (None, None) and coord != pos and name not in self.explored_waypoints]
        
        if not valid_targets:
            self.explored_waypoints.clear() # clear explored sets - reset waypoints
            return self.pick_target_area(pos)
        
        selected_target = None
        if self.config['target_area'] == 'furthest':
            selected_target = max(valid_targets, key=lambda t: get_mahattan_distance(pos, t))
        elif self.config['target_area'] == 'closest':
            selected_target = min(valid_targets, key=lambda t: get_mahattan_distance(pos, t))
        else:
            selected_target = random.choice(valid_targets)
            
        return selected_target
        
    def execute_arrival_logic(self, pos: tuple):
        # just stay
        if self.config['post_arrival'] == 'stay':
            return Move.STAY
            
        if self.config['post_arrival'] == 'wander':
            valid_moves = []
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
                if self.is_tile_safe((nr, nc)):
                    valid_moves.append(m)
            
            if valid_moves:
                return random.choice(valid_moves)
                
            # Failsafe: If the ONLY safe move is backward (e.g. entering a tight corner), take it
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
                if self.is_tile_safe((nr, nc)):
                    return m
                
        if self.config['post_arrival'] == 'loop':
            # print("Loop mode")
            self.explored_waypoints.append(self.target_node)
            # print("Explored: ")
            self.state = GhostState.TRANSIT
            self.target_node = self.pick_target_area(pos)
            
            transit_move = self.get_greedy_transit_move(pos, self.target_node)
            # print("Move chosen: ", transit_move)
                            
             # If we return STAY, we hit a wall/local minimum and greedy routing failed
            if transit_move == Move.STAY:
                self.target_node = self.pick_target_area(pos) # Reroute
                return Move.STAY
            else:
                return transit_move
                    
        return Move.STAY
    
    def get_greedy_transit_move(self, current_pos: tuple, target_pos: tuple) -> Move:
        """Greedily finds a known, safe move that reduces Manhattan distance to the target."""
        best_moves = []
        min_dist = float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = current_pos[0] + move.value[0], current_pos[1] + move.value[1]
            next_pos = (nr, nc)
            
            # STRICTLY check against explored_map == 0 (no -1 assumptions) and avoid dead ends
            if self.is_tile_safe(next_pos):
                dist = get_mahattan_distance(next_pos, target_pos)
                
                # Slight penalty to prevent 180-degree backtracking if other options exist
                if hasattr(self, 'previous_step') and move == self._get_reverse_move(self.previous_step):
                    dist += 5 
                
                if dist < min_dist:
                    min_dist = dist
                    best_moves = [move]
                elif dist == min_dist:
                    best_moves.append(move)
                    
        if best_moves:
            return random.choice(best_moves)
            
        return Move.STAY

    def _get_reverse_move(self, move: Move) -> Move:
        if move == Move.STAY: return Move.STAY
        reverse = {Move.UP: Move.DOWN, Move.DOWN: Move.UP, Move.LEFT: Move.RIGHT, Move.RIGHT: Move.LEFT}
        return reverse.get(move, Move.STAY)
            
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        
        self.map_BFS_cache.clear()
        self.update_explored_map(map_state)
        self.update_dead_ends()
         
        current_vision = self.get_visible_cells(my_position)
        self.update_heatmap(current_vision, enemy_position)
            
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.state = GhostState.TRANSIT
            self.target_node = None
            
            best_move = Move.STAY
            max_eval = -float('inf')
            alpha = -float('inf')
            beta = float('inf')
        
            for move in [Move.STAY, Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            
                if self.is_tile_valid(next_pos):
                    eval = self.minimax(next_pos, enemy_position, map_state, self.max_depth, alpha, beta, False)
                    if eval > max_eval:
                        max_eval = eval
                        best_move = move
                    alpha = max(alpha, eval)
                    
            self.previous_step = best_move
            return best_move
        
        chosen_action = Move.STAY
        
        if self.state == GhostState.OPENING:
            if self.config['first_move'] == 'observe' and step_number <= self.config['observe_duration']:
                chosen_action = Move.STAY
            else:
                self.state = GhostState.TRANSIT
            
        if self.state == GhostState.TRANSIT:
            if self.target_node is None or my_position == self.target_node:
                self.target_node = self.pick_target_area(my_position)
                
            # print("Target: ", self.target_node)
                
            if my_position == self.target_node:
                self.state = GhostState.ARRIVED
            else:
                transit_move = self.get_greedy_transit_move(my_position, self.target_node)
                
                # If we return STAY, we hit a wall/local minimum and greedy routing failed
                if transit_move == Move.STAY:
                    self.target_node = self.pick_target_area(my_position) # Reroute
                    chosen_action = Move.STAY
                else:
                    chosen_action = transit_move
                    
        if self.state == GhostState.ARRIVED:
            chosen_action = self.execute_arrival_logic(my_position)
            # print("Entering arrival logic")
        
        self.previous_step = chosen_action
        # print("Current position: ", my_position)
        # print("Chosen action: ", chosen_action)
        return chosen_action  