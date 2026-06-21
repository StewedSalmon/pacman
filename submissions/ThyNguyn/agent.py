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

from collections import deque
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
        self.path = []
        self.visited = set()
        # TODO: Initialize any data structures you need
        # Examples:
        # - self.path = []  # Store planned path
        # - self.visited = set()  # Track visited positions
        self.name = "Be Pacman De thuong"
        # Memory for limited observation mode
        self.last_known_enemy_pos = None
    
    def step(self, map_state: np.ndarray, 
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
            # No information about enemy - explore randomly
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)
        
        # Example: Simple greedy approach (replace with your algorithm)
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        target = enemy_position or self.last_known_enemy_pos

        if target is None:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                if self._is_valid_move(my_position, move, map_state):
                    return (move, 1)
            return (Move.STAY, 1)

        return self._choose_best_action(my_position, target, map_state)
    
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
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _get_neighbors(self, pos, map_state):
        result = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = self._apply_move(pos, move)
            if self._is_valid_position(nxt, map_state):
                result.append((nxt, move))
        return result

    def _bfs(self, start, goal, map_state):
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()

            if pos == goal:
                return path

            for nxt, move in self._get_neighbors(pos, map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))

        return []
    
    def _choose_best_action(self, my_position, target, map_state):
        best_action = (Move.STAY, 1)
        best_score = None

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            max_steps = self._max_valid_steps(
                my_position,
                move,
                map_state,
                self.pacman_speed
            )

            for steps in range(1, max_steps + 1):
                new_pos = self._simulate_action(my_position, move, steps)
                score = self._evaluate_position(new_pos, target, map_state)

                if best_score is None or score > best_score:
                    best_score = score
                    best_action = (move, steps)

        return best_action


    def _simulate_action(self, pos, move, steps):
        current = pos

        for _ in range(steps):
            current = self._apply_move(current, move)

        return current


    def _evaluate_position(self, pacman_pos, ghost_pos, map_state):
        path = self._bfs(pacman_pos, ghost_pos, map_state)

        if pacman_pos == ghost_pos:
            return 0

        if not path:
            return -999
        #sentinel value nô chọn 

        shortest_dist = len(path)

        mobility = len(self._get_neighbors(pacman_pos, map_state))
        # đứng ô này có bao nhiêu cách đi 

        score = -shortest_dist + 0.1 * mobility
        #từ agent + số cách đi

        return score

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
        self.previous_position = None
    
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
        
        # Update memory if enemy is visible
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        
        # Use current sighting, fallback to last known, or move randomly
        threat = enemy_position or self.last_known_enemy_pos
        
        if threat is None:
            # No information about enemy - move randomly
            best_move = self._wander_move(my_position, map_state)
            self.previous_position = my_position
            return best_move
        
        best_move = Move.STAY
        best_score = None

        for move in [Move.RIGHT, Move.LEFT, Move.UP, Move.DOWN, Move.STAY]:
            if move != Move.STAY and not self._is_valid_move(my_position, move, map_state):
                continue

            next_pos = self._apply_move(my_position, move)
            score = self._score_escape_position(next_pos, threat, map_state)

            if move == Move.STAY:
                score -= 8

            if self.previous_position is not None and next_pos == self.previous_position:
                score -= 4

            if best_score is None or score > best_score:
                best_score = score
                best_move = move

        self.previous_position = my_position
        return best_move
    
    # Helper methods (you can add more)
    
    def _is_valid_move(self, pos: tuple, move: Move, map_state: np.ndarray) -> bool:
        delta_row, delta_col = move.value
        new_pos = (pos[0] + delta_row, pos[1] + delta_col)
        return self._is_valid_position(new_pos, map_state)
    
    def _is_valid_position(self, pos: tuple, map_state: np.ndarray) -> bool:
        row, col = pos
        height, width = map_state.shape
        
        if row < 0 or row >= height or col < 0 or col >= width:
            return False
        
        return map_state[row, col] == 0
    
    def _find_furthest_position(self, my_position, pacman_position, map_state):
        queue = deque([my_position])
        visited = {my_position}

        best_pos = my_position
        best_score = None

        while queue:
            pos = queue.popleft()

            dist_to_pacman = self._shortest_distance(pos, pacman_position, map_state)

            if dist_to_pacman is None:
                dist_to_pacman = 0

            exits = self._count_exits(pos, map_state)
            dist_from_ghost = self._shortest_distance(my_position, pos, map_state)

            if dist_from_ghost is None:
                dist_from_ghost = 0

            score = 0

            # càng xa Pacman càng tốt
            score += dist_to_pacman * 10

            # ưu tiên ô có nhiều đường thoát
            score += exits * 3

            # tránh dead-end
            if exits <= 1:
                score -= 25

            # né Pacman speed 2
            if self._in_speed2_danger(pos, pacman_position, map_state):
                score -= 40

            # không chọn target quá xa Ghost hiện tại
            score -= dist_from_ghost

            if best_score is None or score > best_score:
                best_score = score
                best_pos = pos

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._apply_move(pos, move)

                if nxt not in visited and self._is_valid_position(nxt, map_state):
                    visited.add(nxt)
                    queue.append(nxt)

        return best_pos
    
    def _count_exits(self, pos, map_state):
        return len([
            move for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
            if self._is_valid_move(pos, move, map_state)
        ])

    def _score_escape_position(self, pos, pacman_position, map_state):
        dist_to_pacman = self._shortest_distance(pacman_position, pos, map_state)
        if dist_to_pacman is None:
            dist_to_pacman = self._manhattan_distance(pos, pacman_position)

        exits = self._count_exits(pos, map_state)
        space = self._reachable_count(pos, map_state, max_depth=7)

        score = dist_to_pacman * 12
        score += exits * 5
        score += space

        # hẻm cụt rất dễ bị Pacman speed 2 ép chết
        if exits <= 1:
            score -= 35

        # nếu sau lượt này Pacman có thể lao thẳng tới sát mình thì né mạnh
        if self._in_speed2_danger(pos, pacman_position, map_state):
            score -= 60

        return score

    def _wander_move(self, my_position, map_state):
        best_move = Move.STAY
        best_score = None

        for move in [Move.RIGHT, Move.LEFT, Move.UP, Move.DOWN]:
            if not self._is_valid_move(my_position, move, map_state):
                continue

            next_pos = self._apply_move(my_position, move)
            score = self._reachable_count(next_pos, map_state, max_depth=7)
            score += self._count_exits(next_pos, map_state) * 4

            if self.previous_position is not None and next_pos == self.previous_position:
                score -= 3

            if best_score is None or score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _reachable_count(self, start, map_state, max_depth=6):
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            pos, depth = queue.popleft()
            if depth >= max_depth:
                continue

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._apply_move(pos, move)
                if nxt not in visited and self._is_valid_position(nxt, map_state):
                    visited.add(nxt)
                    queue.append((nxt, depth + 1))

        return len(visited)

    def _bfs_path(self, start, goal, map_state):
        queue = deque([(start, [])])
        visited = {start}

        while queue:
            pos, path = queue.popleft()

            if pos == goal:
                return path

            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nxt = self._apply_move(pos, move)

                if nxt not in visited and self._is_valid_position(nxt, map_state):
                    visited.add(nxt)
                    queue.append((nxt, path + [move]))

        return []

    def _shortest_distance(self, start, goal, map_state):
        if start == goal:
            return 0

        path = self._bfs_path(start, goal, map_state)

        if not path:
            return None

        return len(path)

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _manhattan_distance(self, pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _in_speed2_danger(self, ghost_pos, pacman_pos, map_state):
        gr, gc = ghost_pos
        pr, pc = pacman_pos

        if gr == pr and abs(gc - pc) <= 2:
            step = 1 if gc > pc else -1
            for c in range(pc + step, gc + step, step):
                if map_state[pr, c] == 1:
                    return False
            return True

        if gc == pc and abs(gr - pr) <= 2:
            step = 1 if gr > pr else -1
            for r in range(pr + step, gr + step, step):
                if map_state[r, pc] == 1:
                    return False
            return True

        return False
