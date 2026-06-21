"""
Tuned hide/seek agents for the Pacman vs Ghost arena.

Pacman: A* path finding with path caching, fog memory, and speed-aware sprinting.
Ghost: one-step adversarial scoring that maximizes Pacman's estimated catch time.
"""

import heapq
import itertools
import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


CARDINAL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]


class GridTools:
    """Shared grid helpers. Unseen cells are risky, but still traversable."""

    def _in_bounds(self, pos, map_state):
        row, col = pos
        height, width = map_state.shape
        return 0 <= row < height and 0 <= col < width

    def _is_valid_position(self, pos, map_state):
        return self._in_bounds(pos, map_state) and map_state[pos[0], pos[1]] != 1

    def _is_known_empty(self, pos, map_state):
        return self._in_bounds(pos, map_state) and map_state[pos[0], pos[1]] == 0

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return (pos[0] + dr, pos[1] + dc)

    def _neighbors(self, pos, map_state, include_stay=False):
        moves = CARDINAL_MOVES + ([Move.STAY] if include_stay else [])
        for move in moves:
            nxt = self._apply_move(pos, move)
            if self._is_valid_position(nxt, map_state):
                yield nxt, move

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_distance(self, start, goal, map_state, limit=500):
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}
        expanded = 0

        while queue and expanded < limit:
            pos, dist = queue.popleft()
            expanded += 1

            for nxt, _ in self._neighbors(pos, map_state):
                if nxt in visited:
                    continue
                if nxt == goal:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))

        return None

    def _reachable_count(self, start, map_state, depth=8):
        queue = deque([(start, 0)])
        visited = {start}

        while queue:
            pos, dist = queue.popleft()
            if dist >= depth:
                continue
            for nxt, _ in self._neighbors(pos, map_state):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append((nxt, dist + 1))

        return len(visited)


class PacmanAgent(BasePacmanAgent, GridTools):
    """
    Seeker.

    Tunable knobs:
    - fog_replan_interval: smaller means more active exploration when the ghost is lost.
    - unseen_step_cost: bigger means avoid fog unless needed.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Tuned A* Seeker"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.last_known_enemy_pos = None
        self.last_target = None
        self.path = []
        self.visited_when_lost = {}

        self.fog_replan_interval = 4
        self.unseen_step_cost = 2

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int):
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_target = enemy_position
            self.path = self._a_star(my_position, enemy_position, map_state)
        elif not self.last_known_enemy_pos:
            return self._explore(my_position, map_state, step_number)

        target = enemy_position or self.last_known_enemy_pos

        should_replan = (
            not self.path
            or target != self.last_target
            or step_number % self.fog_replan_interval == 0
        )
        if should_replan:
            self.last_target = target
            self.path = self._a_star(my_position, target, map_state)

        if not self.path:
            self.last_known_enemy_pos = None
            return self._explore(my_position, map_state, step_number)

        move = self.path.pop(0)
        steps = 1
        while steps < self.pacman_speed and self.path and self.path[0] == move:
            self.path.pop(0)
            steps += 1

        valid_steps = self._max_valid_steps(my_position, move, map_state, steps)
        if valid_steps <= 0:
            self.path = []
            return self._explore(my_position, map_state, step_number)
        return (move, valid_steps)

    def _a_star(self, start, goal, map_state):
        tie = itertools.count()
        open_heap = [(self._manhattan(start, goal), next(tie), 0, start, [])]
        best_cost = {start: 0}

        while open_heap:
            _, _, cost, pos, path = heapq.heappop(open_heap)
            if pos == goal:
                return path
            if cost > best_cost.get(pos, float("inf")):
                continue

            for nxt, move in self._neighbors(pos, map_state):
                step_cost = self.unseen_step_cost if map_state[nxt[0], nxt[1]] == -1 else 1
                new_cost = cost + step_cost
                if new_cost >= best_cost.get(nxt, float("inf")):
                    continue
                best_cost[nxt] = new_cost
                priority = new_cost + self._manhattan(nxt, goal)
                heapq.heappush(open_heap, (priority, next(tie), new_cost, nxt, path + [move]))

        return []

    def _explore(self, my_position, map_state, step_number):
        self.visited_when_lost[my_position] = step_number

        best = None
        best_score = -float("inf")
        for move in CARDINAL_MOVES:
            steps = self._max_valid_steps(my_position, move, map_state, self.pacman_speed)
            if steps <= 0:
                continue
            landing = my_position
            for _ in range(steps):
                landing = self._apply_move(landing, move)

            last_seen = self.visited_when_lost.get(landing, -9999)
            unseen_bonus = 8 if map_state[landing[0], landing[1]] == -1 else 0
            space_bonus = self._reachable_count(landing, map_state, depth=5)
            score = (step_number - last_seen) + unseen_bonus + 0.2 * space_bonus
            if score > best_score:
                best_score = score
                best = (move, steps)

        return best or (Move.STAY, 1)

    def _max_valid_steps(self, pos, move, map_state, desired_steps):
        steps = 0
        current = pos
        for _ in range(min(self.pacman_speed, max(1, desired_steps))):
            nxt = self._apply_move(current, move)
            if not self._is_valid_position(nxt, map_state):
                break
            current = nxt
            steps += 1
        return steps


class GhostAgent(BaseGhostAgent, GridTools):
    """
    Hider.

    Tunable knobs:
    - pacman_speed_guess: should match arena --pacman-speed.
    - distance_weight: higher values favor raw distance.
    - space_weight: higher values avoid corridors and dead ends.
    - reverse_penalty: discourages jittering back and forth.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Tuned Evasive Hider"
        self.last_known_enemy_pos = None
        self.previous_position = None

        self.pacman_speed_guess = 2
        self.distance_weight = 10.0
        self.space_weight = 0.7
        self.corner_penalty = 4.0
        self.reverse_penalty = 1.5

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position

        threat = enemy_position or self.last_known_enemy_pos
        candidates = self._candidate_moves(my_position, map_state)

        if threat is None:
            move = self._wander(candidates, my_position, map_state)
        else:
            move = self._best_escape_move(candidates, my_position, threat, map_state)

        self.previous_position = my_position
        return move

    def _candidate_moves(self, my_position, map_state):
        moves = []
        for move in CARDINAL_MOVES + [Move.STAY]:
            nxt = self._apply_move(my_position, move)
            if move == Move.STAY or self._is_known_empty(nxt, map_state):
                moves.append((move, nxt))
        return moves

    def _best_escape_move(self, candidates, my_position, threat, map_state):
        best_move = Move.STAY
        best_score = -float("inf")

        for move, ghost_next in candidates:
            distance = self._bfs_distance(threat, ghost_next, map_state)
            if distance is None:
                distance = self._manhattan(threat, ghost_next) + 12

            pacman_turns = distance / float(self.pacman_speed_guess)
            space = self._reachable_count(ghost_next, map_state, depth=7)
            degree = sum(1 for _ in self._neighbors(ghost_next, map_state))
            is_reverse = self.previous_position is not None and ghost_next == self.previous_position

            score = (
                self.distance_weight * pacman_turns
                + self.space_weight * space
                - self.corner_penalty * max(0, 2 - degree)
                - self.reverse_penalty * int(is_reverse)
            )

            if move == Move.STAY:
                score -= 3

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _wander(self, candidates, my_position, map_state):
        best_move = Move.STAY
        best_score = -float("inf")
        height, width = map_state.shape
        pacman_start_prior = (int(height * 0.75), width // 2)

        for move, nxt in candidates:
            if move == Move.STAY:
                continue
            space = self._reachable_count(nxt, map_state, depth=6)
            distance_from_prior = self._manhattan(nxt, pacman_start_prior)
            top_half_bonus = max(0, height // 2 - nxt[0])
            last_pos_penalty = 2 if self.previous_position is not None and nxt == self.previous_position else 0
            score = (
                space
                + 1.8 * distance_from_prior
                + 1.2 * top_half_bonus
                - last_pos_penalty
                + random.random() * 0.01
            )
            if score > best_score:
                best_score = score
                best_move = move

        return best_move
