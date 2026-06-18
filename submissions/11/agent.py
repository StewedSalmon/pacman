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

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np


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
        # - self.name = "Your Agent Name"
        self.name = "BFS Pacman"
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int):
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Ghost's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move or (Move, steps): Direction to move (optionally with step count)
        """
        # TODO: Implement your search algorithm here
        # =========== DEBUG =================
        # print(f"Step {step_number}: My pos={my_position}, Enemy pos={enemy_position}")
        # ===================================
    
        path = self.bfs(my_position, enemy_position, map_state)

        # =========== DEBUG =================
        # print(f"  Found path: {[m.name for m in path[:5]]}")  # First 5 moves
        # ===================================


        # Ko tìm được đường đi (hoặc BFS trả [STAY]) -> đứng yên
        if not path or path[0] == Move.STAY:
            return Move.STAY

        first_move = path[0]
        if self.pacman_speed <= 1: # Nếu chỉ đi được 1 bước, trả luôn move đầu tiên (path[0], 1)
            return first_move

        # Khi chạy test sẽ chỉnh pacman_speed = 2, nhưng đề phòng trường hợp speed > 2 nên code chung luôn
        # Nếu đầu path có các move trùng nhau liên tiếp, gộp thành (Move, steps)
        # với steps <= pacman_speed và không đi xuyên tường (check bằng _max_valid_steps).
        max_consider = min(len(path), self.pacman_speed)
        steps = 1
        while steps < max_consider and path[steps] == first_move:
            steps += 1

        steps = self._max_valid_steps(my_position, first_move, map_state, steps)
        if steps > 1:
            return (first_move, steps)

        return first_move

    
    
    # Algorithm

    def bfs(self, start, goal, map_state):
        """
        Find shortest path from start to goal using BFS.
        
        Returns:
            List of Move enums representing the path, or [Move.STAY] if no path
        """
        from collections import deque
        
        # Queue stores (position, path_to_reach_it)
        queue = deque([(start, [])])
        visited = {start}
        
        while queue:
            current_pos, path = queue.popleft()
            
            # Found the goal!
            if current_pos == goal:
                return path
            
            # Explore neighbors
            for next_pos, move in self._get_neighbors(current_pos, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        
        # No path found
        return [Move.STAY]



    # Helper methods (you can add more)
    
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

    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

# Ghost agent: Uses BFS to run to the cell furthest from pacman
# Strategy overview:
#   1. Run BFS from pacmans position to compute distances from pacman to every reachable cell.
#   2. Identify the cell with the maximum distance from pacman (safest)
#   3. Filter out dead end targets (cells with only 1 exit) to avoid getting cornered.
#   4. Pick the closest safest cell to the ghost and use BFS again to find the path from ghost to that cell.
#   5. Before committing, do a safety check: if the intended move would place the ghost within Manhattan distance 2 of pacman, reject it.
#   6. Fallback: pick whichever neighboring cell maximises Manhattan distance from pacman, preferring cells with more exits.

class GhostAgent(BaseGhostAgent):
    """
    Ghost (Hider) Agent - Goal: Avoid being caught
    
    Implement your search algorithm to evade Pacman as long as possible.
    Suggested algorithms: BFS (find furthest point), Minimax, Monte Carlo
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass
    
    def step(self, map_state: np.ndarray, 
             my_position: tuple, 
             enemy_position: tuple,
             step_number: int) -> Move:
        """
        Decide the next move.
        
        Args:
            map_state: 2D numpy array where 1=wall, 0=empty
            my_position: Your current (row, col)
            enemy_position: Pacman's current (row, col)
            step_number: Current step number (starts at 1)
            
        Returns:
            Move: One of Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY
        """
        return self.find_furthest_position(my_position, enemy_position, map_state)

    def find_furthest_position(self, my_position, enemy_position, map_state):
        """
        Use BFS from enemy's position to find furthest reachable point.
        Then navigate toward it.
        
        Returns:
            Move enum to move toward furthest position
        """
        from collections import deque
        
        # 1. BFS from enemy to find distances
        queue = deque([(enemy_position, 0)])
        visited = {enemy_position: 0}   # Maps position -> distance from Pacman
        
        while queue:
            current, dist = queue.popleft()
            
            # Expand to all neighbors
            for next_pos, _ in self._get_neighbors(current, map_state):
                if next_pos not in visited:
                    visited[next_pos] = dist + 1
                    queue.append((next_pos, dist + 1))
        
        # 2. Find maximum distance among reachable positions
        max_dist = -1
        best_positions = []
        
        for pos, dist in visited.items():
            if dist > max_dist:
                max_dist = dist
                best_positions = [pos]
            elif dist == max_dist:
                best_positions.append(pos)
        
        # 3. Filter out dead end targets to avoid getting cornered
        safe_positions = [p for p in best_positions if self._count_exits(p, map_state) >= 2]
        # If all max distance cells are dead ends, keep the original list as fallback
        if not safe_positions:
            safe_positions = best_positions
        
        # 4. Navigate to closest of the furthest safe positions
        best_safety_move = Move.STAY
        best_safety_dist = self._manhattan_distance(my_position, enemy_position)
        
        if safe_positions:
            best_target = min(safe_positions, 
                            key=lambda p: self._manhattan_distance(my_position, p))

            path = self.bfs(my_position, best_target, map_state)

            if path:
                intended_move = path[0]
                next_pos = self._apply_move(my_position, intended_move)
                
                # 5. Safety check: only commit if not too close to pacman (Manhattan distance > 2)
                if self._manhattan_distance(next_pos, enemy_position) > 2:
                    # Also check the next cell isnt a dead-end when pacman is nearby
                    if self._count_exits(next_pos, map_state) >= 2 or self._manhattan_distance(next_pos, enemy_position) > 4:
                        return intended_move
        
        # 6. Fallback: greedy strategy preferring cells with more exits
        # score = distance from pacman + bonus for having more escape routes
        best_score = -1
        for next_pos, move in self._get_neighbors(my_position, map_state):
            distance = self._manhattan_distance(next_pos, enemy_position)
            exits = self._count_exits(next_pos, map_state)
            # Combine distance and exit count: distance matters most, exits break ties
            score = distance * 10 + exits
            if score > best_score:
                best_score = score
                best_safety_move = move
                
        return best_safety_move


    # Helper methods
    def _count_exits(self, pos, map_state):
        """Count how many valid moves exist from a position (1 = dead-end, 2+ = open)."""
        return len(self._get_neighbors(pos, map_state))

    def _is_dead_end(self, pos, map_state):
        """Check if a position is a dead-end (only 1 exit)."""
        return self._count_exits(pos, map_state) <= 1

    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        """Check if a move from pos is valid for at least one step."""
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        """Check if position is valid (not wall, within bounds)."""
        row, col = pos
        height, width = map_state.shape
        
        # Check bounds
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        # Check not a wall
        return map_state[row, col] == 0

    def _apply_move(self, pos, move):
        """Apply a move to a position, return new position."""
        delta_row, delta_col = move.value
        return (pos[0] + delta_row, pos[1] + delta_col)

    def _get_neighbors(self, pos, map_state):
        """Get all valid neighboring positions and their moves."""
        neighbors = []
        
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = self._apply_move(pos, move)
            if self._is_valid_position(next_pos, map_state):
                neighbors.append((next_pos, move))
        
        return neighbors

    def _manhattan_distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def bfs(self, start, target, map_state):
        """
        Find shortest path from start to goal using BFS.
        
        Returns:
            List of Move enums representing the path, or [Move.STAY] if no path
        """
        from collections import deque

        # Queue stores (position, path_to_reach_it)
        queue = deque([(start, [])])
        visited = set([start])

        while queue:
            curr, path = queue.popleft()

            # Found the goal!
            if curr == target:
                return path
            
            # Expand neighbors
            for next_pos, move in self._get_neighbors(curr, map_state):
                if next_pos not in visited:
                    visited.add(next_pos)
                    queue.append((next_pos, path + [move]))
        
        # No path found
        return [Move.STAY]