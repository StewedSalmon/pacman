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


class PacmanAgent(BasePacmanAgent):
    """
    Minimax Pacman with BFS evaluation
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Pacman agent.
        Students can set up any data structures they need here.
        """
        super().__init__(**kwargs)
        self.name = "Minimax Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
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
    
    def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, pacman_turn: bool):
        # base case
        if ghost_position == pacman_position:
           return 0 - depth
        if depth == 0: 
            # eval using BFS
            path = self.BFS(map_state, pacman_position, ghost_position)
            if not path:
                return float('inf') # no path means ghost is trapped, best for pacman
            pacman_ticks = self.pacman_tick(path)
            return (pacman_ticks * 10) + len(path) # tie breaker: prefer shorter paths if same ticks
        
        # minimizing for pacman
        if pacman_turn:
            min_eval = float('inf')
            pacman_next_positions = self.get_sprint_options(pacman_position, map_state)
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
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                
                if self._is_valid_position(next_ghost_pos, map_state):
                    eval = self.minimax(next_ghost_pos, pacman_position, map_state, depth - 1, alpha, beta, True)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
    
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
    
    def get_sprint_options(self, pos: tuple, map_state: np.ndarray):
        """
        Generates all valid moves and step counts for Pacman.
        Returns a list of tuples: (Move, step_count, landing_position)
        """
        options = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            delta_row, delta_col = move.value
            current_pos = pos        
            for steps in range(1, self.pacman_speed + 1):
                next_pos = (current_pos[0] + delta_row, current_pos[1] + delta_col)
                # Stop projecting if we hit a wall
                if not self._is_valid_position(next_pos, map_state):
                    break 
                current_pos = next_pos
                options.append((move, steps, current_pos))
                
        return options
    
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
        
        best_move = Move.STAY
        best_steps = 1
        min_eval = float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        possible_moves = self.get_sprint_options(my_position, map_state)
        for move, steps, landing_pos in possible_moves:
            eval = self.minimax(enemy_position, landing_pos, map_state, 3, alpha, beta, False)
            if eval < min_eval:
                min_eval = eval
                best_move = move
                best_steps = steps
            beta = min(beta, eval)
            
        if best_steps > 1:
            return (best_move, best_steps)
        return (best_move, 1)

    def _explore(self, my_position: tuple, map_state: np.ndarray):
        """Random exploration when enemy position is unknown."""
        all_moves = self.get_sprint_options(my_position, map_state)
        if all_moves:
            move, steps, _ = random.choice(all_moves)
            if steps > 1:
                return (move, steps)
            return (move, 1)
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
        """
        Initialize the Ghost agent.
        """
        super().__init__(**kwargs)
        self.name = "Minimax Ghost"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
        self.pacman_speed = 2
    
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
    
    def minimax(self, ghost_position: tuple, pacman_position: tuple, map_state: np.ndarray, depth: int, alpha: int, beta: int, ghost_turn: bool):
        # base case
        if depth == 0 or ghost_position == pacman_position:
            # eval using BFS
            path = self.BFS(map_state, ghost_position, pacman_position)
            if not path:
                return 0
            pacman_ticks = self.pacman_tick(path)
            return (pacman_ticks * 10) + len(path) # tie breaker: prefer shorter paths if same ticks
        # maximizing for ghost
        if ghost_turn:
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_ghost_pos = (ghost_position[0] + move.value[0], ghost_position[1] + move.value[1])
                
                if self._is_valid_position(next_ghost_pos, map_state):
                    eval = self.minimax(next_ghost_pos, pacman_position, map_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval)
                    alpha = max(alpha, eval)
                    if beta <= alpha:
                        break
            return max_eval
        
        # minimizing for pacman
        else:
            min_eval = float('inf')
            pacman_next_positions = self.pacman_simulated_positions(pacman_position, map_state)
            for next_pacman_pos in pacman_next_positions:
                eval = self.minimax(ghost_position, next_pacman_pos, map_state, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
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
    
    def pacman_simulated_positions(self, pacman_position: tuple, map_state: np.ndarray):      
        """Generate sensible next moves for Pacman based on current position and map state."""
        simulated_positions = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            row, col = move.value
            current_position = pacman_position
            
            for step in range(self.pacman_speed):
                next_position = (current_position[0] + row, current_position[1] + col)
                
                if not self._is_valid_position(next_position, map_state):
                    break
                current_position = next_position
                
            if current_position != pacman_position:
                simulated_positions.append(current_position)
        return simulated_positions
            
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            return self._random_move(my_position, map_state)
        
        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
            
            if self._is_valid_position(next_pos, map_state):
                eval = self.minimax(next_pos, threat, map_state, 3, alpha, beta, False)
                
                if eval > max_eval:
                    max_eval = eval
                    best_move = move
                    
                alpha = max(alpha, eval)
        
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
