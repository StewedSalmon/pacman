import sys
from pathlib import Path

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))
import heapq
import queue
import itertools
from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random

class PacmanAgent(BasePacmanAgent):
    """
    Pacman agent implementing A* Search for optimal pathfinding.
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "A-Star Seeker"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self.current_path = []
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        
        # 1. Update Memory
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            # If we see the enemy, force a recalculation of the path
            self.current_path = self._a_star_search(my_position, enemy_position, map_state)
            
        target = enemy_position or self.last_known_enemy_pos
        
        # 2. Handle Fog of War (Target Unknown)
        if target is None:
            return self._explore(my_position, map_state)
            
        # 3. Follow the calculated path
        if not self.current_path:
            # If path is empty but we have a target, try recalculating (e.g., we reached the last known pos)
            if my_position != target:
                self.current_path = self._a_star_search(my_position, target, map_state)
            
            # If still empty, we either reached the last known spot or are trapped
            if not self.current_path:
                self.last_known_enemy_pos = None # Clear memory and explore
                return self._explore(my_position, map_state)

        # 4. Extract the next move and calculate maximum sprint distance
        next_move = self.current_path.pop(0)
        
        # Optimize: If we have speed > 1, check how many contiguous steps we can take in this direction 
        # that are also on our A* path.
        steps_to_take = 1
        while (steps_to_take < self.pacman_speed and 
               self.current_path and 
               self.current_path[0] == next_move):
            steps_to_take += 1
            self.current_path.pop(0) # Consume the extra step from our planned path
            
        # Double check validity just in case
        valid_steps = self._max_valid_steps(my_position, next_move, map_state, steps_to_take)
        
        if valid_steps > 0:
            return (next_move, valid_steps)
        return (Move.STAY, 1)

    def _a_star_search(self, start: tuple, goal: tuple, map_state: np.ndarray) -> list:
        """Standard A* Search returning a list of Move enums."""
        
        # We use a tie_breaker counter so heapq never tries to compare Move enums
        tie_breaker = itertools.count()
        
        # Priority Queue elements: (f_score, tie_breaker_id, g_score, current_pos, path_of_moves)
        open_set = []
        heapq.heappush(open_set, (0, next(tie_breaker), 0, start, []))
        visited = set()
        
        while open_set:
            # Unpack the 5 elements, ignoring the tie_breaker variable with '_'
            f, _, g, current, path = heapq.heappop(open_set)
            
            if current == goal:
                return path
                
            if current in visited:
                continue
            visited.add(current)
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (current[0] + dr, current[1] + dc)
                
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    new_g = g + 1
                    h = self._manhattan_distance(next_pos, goal)
                    new_f = new_g + h
                    
                    # Push the next sequence number into the tuple
                    heapq.heappush(open_set, (new_f, next(tie_breaker), new_g, next_pos, path + [move]))
                    
        return [] # No path found

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Fallback random exploration."""
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        for move in all_moves:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps > 0:
                return (move, steps)
        return (Move.STAY, 1)

    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        return map_state[row, col] != 1 # Important: Treat -1 (unseen) as traversable!

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
    
class GhostAgent(BaseGhostAgent):
    """
    BFS-based Ghost agent
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Ghost agent.
        """
        super().__init__(**kwargs)
        self.name = "BFS Ghost"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    def BFS(self, map_state: np.ndarray, start: tuple, goal: tuple) -> list:
        """Perform BFS to find the first move towards goal"""
        queue = [(start, [])]
        visited = set()
        while queue:
            current, path = queue.pop(0)
            if current == goal:
                return path
            
            if current in visited:
                continue
            visited.add(current)
            
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                x, y = move.value
                next_pos = (current[0] + x, current[1] + y)
                
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    queue.append((next_pos, path + [move]))
                    
        return [] # no path found
    
    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        # Update memory with current enemy position
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            
        # Identify threat position
        threat_pos = enemy_position or self.last_known_enemy_pos
        
        # If there's no info on where the enemy is, explore randomly
        if threat_pos is None:
            return self._explore(my_position, map_state)
        
        # Use BFS
        # Scan the map to find all valid empty tiles
        empty_tiles = []
        for r in range(map_state.shape[0]):
            for c in range(map_state.shape[1]):
                if map_state[r, c] == 0: # empty tile
                    empty_tiles.append((r, c))
                    
        # Find the tile furthest from the threat
        target_tile = max(empty_tiles, key=lambda pos: self._manhattan_distance(pos, threat_pos))
        
        # Run BFS to find the first move towards the target tile
        path = self.BFS(map_state, my_position, target_tile)
        if path:
            return path[0] # Return the first move in the path
        else:
            return self._explore(my_position, map_state) # If no path found, explore randomly
        
    def _manhattan_distance(self, pos1: tuple, pos2: tuple) -> int:
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Fallback random exploration."""
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
