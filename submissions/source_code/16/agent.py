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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
from collections import deque

class PacmanAgent(BasePacmanAgent):
    """
    Pacman (Seeker) Agent - Goal: Catch the Ghost
    
    Implement your search algorithm to find and catch the ghost.
    Suggested algorithms: BFS, DFS, A*, Greedy Best-First
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Template Pacman"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray,  # type: ignore
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty, -1=unseen (fog)
            my_position: Your current (row, col) in absolute coordinates
            enemy_position: Ghost's (row, col) if visible, None otherwise
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or explore
        target = enemy_position or self.last_known_enemy_pos
        
        if target is None:
            frontiers = self.find_frontiers(map_state)
            if (frontiers):
                paths = [self.bfs_pacman(map_state, my_position, f) for f in frontiers]
                paths = [p for p in paths if p]
                if (paths):
                    path = min(paths, key=len)
                    target = path[-1]

        if target is not None:
            path = self.bfs_pacman(map_state, my_position, target)
            if path and len(path) > 1:
                next_pos = path[1]
                move = self.pos_to_move(my_position, next_pos)
                steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
                return (move, steps)

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            if self._is_valid_move(my_position, move, map_state):
                return (move, 1)
        return (Move.STAY, 1)
    
    def bfs_pacman(self, map_state, start, goal):
        queue = deque([[start]])
        visited = set([start])
        while queue:
            s = queue.popleft()
            cur = s[-1]
            if cur == goal:
                return s
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (cur[0] + dr, cur[1] + dc)
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(s + [next_pos])
        return None
    # Helper methods (you can add more)
    
    def pos_to_move(self, cur, next_pos):
        dr = next_pos[0] - cur[0]
        dc = next_pos[1] - cur[1]
        if dr == -1: return Move.UP
        if dr == 1: return Move.DOWN
        if dc == -1: return Move.LEFT
        if dc == 1: return Move.RIGHT
        return Move.STAY

    def find_frontiers(self, map_state: np.ndarray):
        frontiers = []
        height, width = map_state.shape
        for r in range(height):
            for c in range(width):
                if(map_state[r][c] == 0):
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if(map_state[nr, nc] == -1):
                                frontiers.append((r, c))
                                break
        return frontiers
    

    def _choose_action(self, pos: tuple, moves, map_state: np.ndarray, desired_steps: int):
        for move in moves:
            max_steps = min(self.pacman_speed, max(1, desired_steps))
            steps = self._max_valid_steps(pos, move, map_state, max_steps)
            if steps > 0:
                return (move, steps)
        return None

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
        """Check if a move from pos is valid for at least one step."""
        return self._max_valid_steps(pos, move, map_state, 1) == 1
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if a position is valid (not a wall and within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0


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
    
    def step(self, map_state: np.ndarray,  # type: ignore
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
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            frontiers = self.find_frontiers(map_state)
            if (frontiers):
                paths = [self.bfs_ghost(map_state, my_position, f) for f in frontiers]
                paths = [p for p in paths if p]
                if (paths):
                    path = max(paths, key=len)
                    next_pos = path[1]
                    return self.pos_to_move(my_position, next_pos)
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return move
            return Move.STAY

        best_move = Move.STAY
        best_dist = -1
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    dr, dc = move.value
                    new_pos = (my_position[0] + dr, my_position[1] + dc)
                    dist = abs(new_pos[0] - threat[0]) + abs(new_pos[1] - threat[1])
                    if dist > best_dist:
                        best_dist = dist
                        best_move = move
        return best_move

    def bfs_ghost(self, map_state, start, goal):
        queue = deque([[start]])
        visited = set([start])
        while queue:
            s = queue.popleft()
            cur = s[-1]
            if cur == goal:
                return s
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                next_pos = (cur[0] + dr, cur[1] + dc)
                if self._is_valid_position(next_pos, map_state) and next_pos not in visited:
                    visited.add(next_pos)
                    queue.append(s + [next_pos])
        return None
    # Helper methods (you can add more)
    
    def pos_to_move(self, cur, next_pos):
        dr = next_pos[0] - cur[0]
        dc = next_pos[1] - cur[1]
        if dr == -1: return Move.UP
        if dr == 1: return Move.DOWN
        if dc == -1: return Move.LEFT
        if dc == 1: return Move.RIGHT
        return Move.STAY


    def find_frontiers(self, map_state: np.ndarray):
        frontiers = []
        height, width = map_state.shape
        for r in range(height):
            for c in range(width):
                if(map_state[r][c] == 0):
                    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < height and 0 <= nc < width:
                            if(map_state[nr, nc] == -1):
                                frontiers.append((r, c))
                                break
        return frontiers


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
