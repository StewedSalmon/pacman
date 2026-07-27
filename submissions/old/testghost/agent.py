"""
TestGhost - Ghost that survives longer than ghost 11 (benchmark: 10 steps)

Analyzed map and game traces:
- Ghost starts (9,10), Pacman starts (15,10) 
- Pacman goes LEFT then UP initially (patrol pattern)
- Ghost 11 survives 10 steps by going RIGHT then UP along right corridors
- Our ghost needs to do BETTER than 10 steps

Key strategies:
1. Opening: Go RIGHT immediately (like ghost 11, away from pacman's LEFT patrol)
   Then immediately turn into corridors to break line-of-sight
2. BFS farthest-point chase that prioritizes cells behind walls
3. Never stay on a straight corridor that pacman can sprint along
4. Use the entire map — run circuits through the outer hallways
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np
import random
from collections import deque
from heapq import heappush, heappop


class PacmanAgent(BasePacmanAgent):
    """Minimal Pacman — placeholder."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Test Pacman (Minimal)"
        self.learned_map = None
        self.last_known_enemy_pos = None

    def step(self, map_state, my_position, enemy_position, step_number):
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
        target = enemy_position or self.last_known_enemy_pos
        if target is None:
            return self._explore(my_position)
        path = self._astar(my_position, target)
        if path and path[0] != Move.STAY:
            return self._execute(my_position, path)
        return self._explore(my_position)

    def _astar(self, start, goal):
        frontier = [(0, start, [])]
        visited = set()
        while frontier:
            _, pos, path = heappop(frontier)
            if pos == goal:
                return path
            if pos in visited:
                continue
            visited.add(pos)
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
                h, w = self.learned_map.shape
                if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0 and (nr, nc) not in visited:
                    new_path = path + [m]
                    heappush(frontier, (len(new_path) + abs(nr - goal[0]) + abs(nc - goal[1]), (nr, nc), new_path))
        return [Move.STAY]

    def _execute(self, pos, path):
        first = path[0]
        steps = 1
        for m in path[1:self.pacman_speed]:
            if m == first: steps += 1
            else: break
        current = pos
        actual = 0
        for _ in range(steps):
            nr, nc = current[0] + first.value[0], current[1] + first.value[1]
            h, w = self.learned_map.shape
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                actual += 1; current = (nr, nc)
            else: break
        return (first, max(1, actual))

    def _explore(self, pos):
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
            h, w = self.learned_map.shape
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                return (m, 1)
        return (Move.STAY, 1)


class GhostAgent(BaseGhostAgent):
    """
    Marathon Ghost — designed to out-survive ghost 11 by using the entire map.
    
    Core principle: maximize the BFS path distance from pacman (not Manhattan).
    When pacman can sprint 2 tiles per turn on straight corridors, the ghost
    must always be around a corner or behind a wall.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Marathon Ghost"
        self.learned_map = None
        self.last_known_enemy_pos = None
        self.assumed_pacman_speed = 2
        self.last_seen_step = 0
        self.panic_timer = 0
        self.prev_positions = []
        self.last_move_dir = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:

        # Map learning
        if self.learned_map is None:
            self.learned_map = np.copy(map_state)
        else:
            visible = map_state != -1
            self.learned_map[visible] = map_state[visible]

        self.prev_positions.append(my_position)
        if len(self.prev_positions) > 20:
            self.prev_positions.pop(0)

        # Enemy tracking
        if enemy_position is not None:
            if (self.last_known_enemy_pos is not None and
                    self.last_seen_step == step_number - 1):
                speed = (abs(enemy_position[0] - self.last_known_enemy_pos[0]) +
                         abs(enemy_position[1] - self.last_known_enemy_pos[1]))
                if 1 <= speed <= 5:
                    self.assumed_pacman_speed = max(self.assumed_pacman_speed, speed)
            self.last_known_enemy_pos = enemy_position
            self.last_seen_step = step_number
            self.panic_timer = 15
        elif self.panic_timer > 0:
            self.panic_timer -= 1

        enemy = enemy_position or self.last_known_enemy_pos

        # Step 1: opening move — rush right (away from typical pacman patrol)
        if step_number == 1:
            if self._is_valid(my_position, Move.RIGHT):
                self.last_move_dir = Move.RIGHT
                return Move.RIGHT

        # If enemy known and in panic mode, evade
        if enemy is not None and self.panic_timer > 0:
            move = self._evade(my_position, enemy, step_number)
        else:
            # No enemy info: keep moving, explore, stay near walls
            move = self._explore_away_from_center(my_position, step_number)

        self.last_move_dir = move
        return move

    def _evade(self, my_pos, enemy_pos, step_number):
        """Find the best escape move using multi-criteria scoring."""
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        # Pre-compute BFS distances from pacman
        pac_dists = self._bfs_distances(enemy_pos)

        best_move = Move.STAY
        best_score = float('-inf')

        for next_pos, move in neighbors:
            score = 0

            # 1. BFS distance from pacman (most important)
            bfs_d = pac_dists.get(next_pos, 0)
            pacman_turns = bfs_d / self.assumed_pacman_speed
            score += pacman_turns * 80

            # 2. Reachable cells (escape routes)
            reachable = self._count_reachable(next_pos, 4)
            score += reachable * 5

            # 3. Dead-end penalty
            exits = len(self._get_neighbors(next_pos))
            if exits == 0:
                score -= 10000
            elif exits == 1:
                score -= 600

            # 4. Line-of-sight penalty
            if self._is_in_los(next_pos, enemy_pos):
                los_d = abs(next_pos[0] - enemy_pos[0]) + abs(next_pos[1] - enemy_pos[1])
                if los_d <= self.assumed_pacman_speed:
                    score -= 500  # Can be reached in 1 sprint
                elif los_d <= self.assumed_pacman_speed * 2:
                    score -= 200  # 2 sprints
                else:
                    score -= 50   # Visible but far

            # 5. Direction change bonus (serpentine movement)
            if self.last_move_dir is not None and move != self.last_move_dir:
                score += 20

            # 6. Anti-oscillation
            visits = sum(1 for p in self.prev_positions[-8:] if p == next_pos)
            score -= visits * 25

            # 7. Prefer staying behind walls (not on long open corridors)
            corridor_exposure = self._corridor_exposure(next_pos, move)
            score -= corridor_exposure * 15

            # 8. Prefer upper/outer parts of map (farther from typical pacman spawn)
            h = self.learned_map.shape[0]
            upper_bonus = max(0, h // 2 - next_pos[0]) * 2
            score += upper_bonus

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _explore_away_from_center(self, my_pos, step_number):
        """When no enemy seen, move toward the outer corridors and keep
        exploring fog areas. Avoid the center of the map."""
        neighbors = self._get_neighbors(my_pos)
        if not neighbors:
            return Move.STAY

        h, w = self.learned_map.shape
        center_r, center_c = h // 2, w // 2

        best_move = Move.STAY
        best_score = float('-inf')

        for next_pos, move in neighbors:
            # Distance from center (prefer outer areas)
            dist_from_center = abs(next_pos[0] - center_r) + abs(next_pos[1] - center_c)

            # Reachable cells
            reachable = self._count_reachable(next_pos, 3)

            # Fog exploration
            has_fog = False
            for d in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = next_pos[0] + d.value[0], next_pos[1] + d.value[1]
                if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == -1:
                    has_fog = True
                    break

            # Anti-oscillation
            visits = sum(1 for p in self.prev_positions[-4:] if p == next_pos)

            # Continue in same direction to cover ground
            momentum = 10 if move == self.last_move_dir else 0

            score = (dist_from_center * 3 + reachable * 2 + 
                     (15 if has_fog else 0) + momentum - visits * 20)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _corridor_exposure(self, pos, direction):
        """Count how many cells are open in the same direction (exposure to sprint)."""
        count = 0
        current = pos
        for _ in range(5):
            nr = current[0] + direction.value[0]
            nc = current[1] + direction.value[1]
            h, w = self.learned_map.shape
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                count += 1
                current = (nr, nc)
            else:
                break
        return count

    # ═══════════════ BFS Utilities ═══════════════

    def _bfs_distances(self, start):
        dists = {start: 0}
        queue = deque([start])
        while queue:
            cur = queue.popleft()
            d = dists[cur]
            for npos, _ in self._get_neighbors(cur):
                if npos not in dists:
                    dists[npos] = d + 1
                    queue.append(npos)
        return dists

    def _count_reachable(self, pos, depth):
        visited = {pos}
        frontier = [pos]
        for _ in range(depth):
            nxt = []
            for cur in frontier:
                for npos, _ in self._get_neighbors(cur):
                    if npos not in visited:
                        visited.add(npos)
                        nxt.append(npos)
            frontier = nxt
        return len(visited) - 1

    def _is_in_los(self, pos1, pos2):
        if pos1[0] != pos2[0] and pos1[1] != pos2[1]:
            return False
        r1, c1 = pos1
        r2, c2 = pos2
        if r1 == r2:
            lo, hi = min(c1, c2), max(c1, c2)
            for c in range(lo + 1, hi):
                if self.learned_map[r1, c] != 0:
                    return False
            return True
        else:
            lo, hi = min(r1, r2), max(r1, r2)
            for r in range(lo + 1, hi):
                if self.learned_map[r, c1] != 0:
                    return False
            return True

    def _get_neighbors(self, pos):
        result = []
        h, w = self.learned_map.shape
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0:
                result.append(((nr, nc), move))
        return result

    def _is_valid(self, pos, move):
        nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
        h, w = self.learned_map.shape
        return 0 <= nr < h and 0 <= nc < w and self.learned_map[nr, nc] == 0
