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
            
    #Predict where ghost maybe move
    def predict_ghost(self, ghost_pos, pacman_pos, map_state):
        best_pos = ghost_pos
        best_score = -1
        for nxt in self.get_neighbors(ghost_pos, map_state):
            dist = self.heuristic(nxt, pacman_pos)

            if dist > best_score:
                best_score = dist
                best_pos = nxt

        return best_pos
    
    #Predict the way ghost maybe move
    def predict_path(self, ghost_pos, pacman_pos, map_state, steps=8):
        path = []
        cur = ghost_pos

        for _ in range(steps):
            cur = self.predict_ghost(cur, pacman_pos, map_state)
            path.append(cur)

        return path

    #find the location to block ghost
    def find_intercept(self, pacman_pos, ghost_path, pac_speed, ghost_speed, bfs_pac):
        for t, cell in enumerate(ghost_path):
            pac_time = bfs_pac[cell[0]][cell[1]] / pac_speed
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
        
        bfs_cost = self.bfs(my_position, map_state)
        ghost_path = self.predict_path(target, my_position, map_state, steps=10)
        intercept = self.find_intercept(my_position, ghost_path, 
                                        self.pacman_speed, 1, bfs_cost)
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
        """
        Simple evasive strategy: move away from Pacman.
        
        When enemy_position is None (limited observation mode),
        uses last known position or moves randomly.
        
        Students should implement better search algorithms like:
        - BFS to find furthest point
        - A* to plan escape route
        - Minimax for adversarial search
        - etc.
        """
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            return self._random_move(my_position, map_state)
        
        # Calculate direction away from threat
        row_diff = my_position[0] - threat[0]
        col_diff = my_position[1] - threat[1]
        
        # List of possible moves in order of preference
        moves = []
        
        # Prioritize vertical movement away from Pacman
        if row_diff > 0:
            moves.append(Move.DOWN)
        elif row_diff < 0:
            moves.append(Move.UP)
        
        # Prioritize horizontal movement away from Pacman
        if col_diff > 0:
            moves.append(Move.RIGHT)
        elif col_diff < 0:
            moves.append(Move.LEFT)
        
        # Try each move in order
        for move in moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            # Check if move is valid
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no preferred move is valid, try any valid move
        all_moves = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
        random.shuffle(all_moves)
        
        for move in all_moves:
            delta_row, delta_col = move.value
            new_pos = (my_position[0] + delta_row, my_position[1] + delta_col)
            
            if self._is_valid_position(new_pos, map_state):
                return move
        
        # If no move is valid, stay
        return Move.STAY

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
