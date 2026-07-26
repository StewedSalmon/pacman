"""Blind-adversary agents for the Pacman vs Ghost arena.

The implementation keeps a map/belief memory for partial observability and
uses short-horizon adversarial action scoring.  It uses only the standard
library and stays comfortably below the one-second turn limit on a 21x21 map.
"""

from collections import deque
from pathlib import Path
import random
import sys
import time

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


CARDINAL = (Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT)
GHOST_MOVES = CARDINAL + (Move.STAY,)


class _MapMemory:
    """Shared graph, distance, visibility, and belief-state utilities."""

    def _memory_init(self):
        self.open_cells = set()
        self.neighbors = {}
        self.distance_cache = {}
        self.visit_count = {}
        self.previous_position = None
        self.enemy_belief = set()
        self.last_seen_enemy = None
        self.last_seen_step = None
        # A blind deterministic route is learnable by opposing patrol agents.
        # SystemRandom makes symmetric choices genuinely mixed while all
        # safety-critical decisions remain deterministic.
        self.rng = random.SystemRandom()
        self.enemy_start_region = None
        self.turns_without_sighting = 0
        self.recent_positions = deque(maxlen=14)

    def _learn_map(self, map_state):
        # In the supplied arena walls are structural knowledge. Unknown (-1)
        # cells are retained as possible paths, but a visible wall is never
        # crossed.
        current_open = {
            (r, c)
            for r in range(map_state.shape[0])
            for c in range(map_state.shape[1])
            if map_state[r, c] != 1
        }
        if current_open == self.open_cells:
            return
        self.open_cells = current_open
        self.neighbors = {}
        for pos in self.open_cells:
            adjacent = []
            for move in CARDINAL:
                nxt = self._apply(pos, move)
                if nxt in self.open_cells:
                    adjacent.append((nxt, move))
            self.neighbors[pos] = adjacent
        self.distance_cache.clear()

    @staticmethod
    def _apply(pos, move, steps=1):
        dr, dc = move.value
        return pos[0] + dr * steps, pos[1] + dc * steps

    def _distances(self, source):
        cached = self.distance_cache.get(source)
        if cached is not None:
            return cached
        distances = {source: 0}
        queue = deque([source])
        while queue:
            pos = queue.popleft()
            for nxt, _ in self.neighbors.get(pos, ()):
                if nxt not in distances:
                    distances[nxt] = distances[pos] + 1
                    queue.append(nxt)
        self.distance_cache[source] = distances
        return distances

    def _distance(self, a, b):
        return self._distances(a).get(b, 10_000)

    def _first_move_toward(self, start, goal):
        if start == goal:
            return Move.STAY
        target_dist = self._distance(start, goal)
        choices = [
            (self._distance(nxt, goal), move)
            for nxt, move in self.neighbors.get(start, ())
        ]
        if not choices:
            return Move.STAY
        choices.sort(key=lambda item: (item[0], item[1].name))
        return choices[0][1] if choices[0][0] < target_dist else Move.STAY

    def _visible_empty(self, map_state):
        return {
            (r, c)
            for r in range(map_state.shape[0])
            for c in range(map_state.shape[1])
            if map_state[r, c] == 0
        }

    def _update_belief(self, map_state, enemy_position, step_number):
        visible = self._visible_empty(map_state)
        if enemy_position is not None:
            self.enemy_belief = {enemy_position}
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number
            self.turns_without_sighting = 0
            return

        self.turns_without_sighting += 1
        if not self.enemy_belief:
            # Arena stochastic starts also preserve the role-dependent halves:
            # Ghost starts in the upper 40%, Pacman in the lower 40%.  Using
            # this public reset rule gives a useful prior without hard-coding
            # one coordinate or breaking stochastic-start evaluation.
            belief = set(self.enemy_start_region or self.open_cells)
        else:
            belief = set()
            for pos in self.enemy_belief:
                belief.add(pos)  # The opponent may stay.
                belief.update(nxt for nxt, _ in self.neighbors.get(pos, ()))
        # Not seeing the opponent is useful negative information.
        self.enemy_belief = belief - visible
        if not self.enemy_belief:
            self.enemy_belief = set(self.open_cells) - visible

    def _degree(self, pos):
        return len(self.neighbors.get(pos, ()))

    def _local_space(self, start, depth=5):
        seen = {start}
        queue = deque([(start, 0)])
        while queue:
            pos, d = queue.popleft()
            if d >= depth:
                continue
            for nxt, _ in self.neighbors.get(pos, ()):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, d + 1))
        return len(seen)


class PacmanAgent(BasePacmanAgent, _MapMemory):
    """Fast seeker: belief search while blind, minimax interception on sight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._memory_init()
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.search_target = None

    def _actions(self, position):
        result = []
        for move in CARDINAL:
            current = position
            for steps in range(1, self.pacman_speed + 1):
                nxt = self._apply(current, move)
                if nxt not in self.open_cells:
                    break
                current = nxt
                result.append((move, steps, current))
        if not result:
            result.append((Move.STAY, 1, position))
        return result

    def _chase_action(self, my_position, enemy_position):
        enemy_next = {enemy_position}
        enemy_next.update(nxt for nxt, _ in self.neighbors.get(enemy_position, ()))
        best = None
        for move, steps, landing in self._actions(my_position):
            distances = sorted(self._distance(landing, pos) for pos in enemy_next)
            captures = sum(d < 2 for d in distances)
            # Optimize worst-case interception, then average distance. Prefer
            # full-speed actions only as a final tie-break.
            score = (
                -captures,
                max(distances),
                sum(distances),
                -self._degree(landing),
                -steps,
                move.name,
            )
            if best is None or score < best[0]:
                best = score, move, steps
        return best[1], best[2]

    def _search_target(self, my_position):
        belief = self.enemy_belief or self.open_cells
        # Candidate targets combine likely enemy cells with under-visited
        # junctions. Sampling keeps the bound independent of map size.
        candidates = list(belief)
        if len(candidates) > 80:
            candidates = sorted(
                candidates,
                key=lambda p: (self.visit_count.get(p, 0), self._distance(my_position, p))
            )[:80]

        best_target = my_position
        best_score = float("-inf")
        for target in candidates:
            dist = self._distance(my_position, target)
            if dist >= 10_000:
                continue
            nearby_belief = sum(
                1 for pos in belief
                if abs(pos[0] - target[0]) + abs(pos[1] - target[1]) <= 5
                and (pos[0] == target[0] or pos[1] == target[1])
            )
            score = (
                4.0 * nearby_belief
                - 1.4 * dist
                - 7.0 * self.visit_count.get(target, 0)
                + 0.8 * self._degree(target)
            )
            if score > best_score:
                best_score, best_target = score, target
        return best_target

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int):
        self._learn_map(map_state)
        if self.enemy_start_region is None:
            cutoff = map_state.shape[0] * 0.4
            self.enemy_start_region = {p for p in self.open_cells if p[0] < cutoff}
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1
        self._update_belief(map_state, enemy_position, step_number)

        if enemy_position is not None:
            self.search_target = None
            move, steps = self._chase_action(my_position, enemy_position)
        else:
            # Commit to a waypoint. Re-selecting a probabilistic target every
            # turn caused short oscillations and left the upper maze unswept.
            if self.search_target is None or my_position == self.search_target:
                self.search_target = self._search_target(my_position)
            target = self.search_target
            move = self._first_move_toward(my_position, target)
            steps = 1
            # Sprint through a known straight corridor when it still reduces
            # distance to the search target.
            current = my_position
            for candidate_steps in range(1, self.pacman_speed + 1):
                nxt = self._apply(current, move)
                if nxt not in self.open_cells:
                    break
                if self._distance(nxt, target) <= self._distance(current, target):
                    steps = candidate_steps
                    current = nxt
                else:
                    break

        self.previous_position = my_position
        return (move, max(1, min(steps, self.pacman_speed)))


class GhostAgent(BaseGhostAgent, _MapMemory):
    """Risk-aware hider using robust scoring against Pacman's speed-2 moves."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._memory_init()
        # The assignment fixes seeker speed at two; kwargs are not supplied to
        # GhostAgent by this arena.
        self.pacman_speed = 2
        self.search_deadline = 0.0
        self.search_cache = {}

    def _pacman_landings(self, position):
        result = {position}
        for move in CARDINAL:
            current = position
            for _ in range(self.pacman_speed):
                nxt = self._apply(current, move)
                if nxt not in self.open_cells:
                    break
                result.add(nxt)
                current = nxt
        return result

    def _update_pacman_belief(self, map_state, enemy_position, step_number):
        """Exact Pacman transition model, including straight speed-2 actions."""
        visible = self._visible_empty(map_state)
        if enemy_position is not None:
            self.enemy_belief = {enemy_position}
            self.last_seen_enemy = enemy_position
            self.last_seen_step = step_number
            self.turns_without_sighting = 0
            return

        self.turns_without_sighting += 1
        if not self.enemy_belief:
            belief = set(self.enemy_start_region or self.open_cells)
        else:
            belief = set()
            for pos in self.enemy_belief:
                belief.update(self._pacman_landings(pos))
        self.enemy_belief = belief - visible
        if not self.enemy_belief:
            self.enemy_belief = set(self.open_cells) - visible

    def _line_visible(self, a, b, radius=5):
        if abs(a[0] - b[0]) + abs(a[1] - b[1]) > radius:
            return False
        if a[0] == b[0]:
            lo, hi = sorted((a[1], b[1]))
            return all((a[0], c) in self.open_cells for c in range(lo, hi + 1))
        if a[1] == b[1]:
            lo, hi = sorted((a[0], b[0]))
            return all((r, a[1]) in self.open_cells for r in range(lo, hi + 1))
        return False

    def _ghost_landings(self, position):
        result = {position}
        result.update(nxt for nxt, _ in self.neighbors.get(position, ()))
        return result

    def _state_value(self, ghost_pos, pacman_pos):
        distance = self._distance(ghost_pos, pacman_pos)
        visible_penalty = 18 if self._line_visible(ghost_pos, pacman_pos) else 0
        return (
            distance * 20
            + self._degree(ghost_pos) * 5
            + self._local_space(ghost_pos, 4)
            - visible_penalty
            - self.recent_positions.count(ghost_pos) * 3
        )

    def _adversarial_value(self, ghost_pos, pacman_pos, depth):
        """Max-min search over complete simultaneous turns."""
        if time.perf_counter() >= self.search_deadline:
            raise TimeoutError
        if abs(ghost_pos[0] - pacman_pos[0]) + abs(ghost_pos[1] - pacman_pos[1]) < 2:
            return -100_000 - depth
        if depth <= 0:
            return self._state_value(ghost_pos, pacman_pos)

        key = (ghost_pos, pacman_pos, depth)
        cached = self.search_cache.get(key)
        if cached is not None:
            return cached

        best = -float("inf")
        ghost_next = sorted(
            self._ghost_landings(ghost_pos),
            key=lambda g: self._distance(g, pacman_pos),
            reverse=True,
        )
        pacman_next = sorted(
            self._pacman_landings(pacman_pos),
            key=lambda p: self._distance(ghost_pos, p),
        )
        for next_ghost in ghost_next:
            worst = float("inf")
            for next_pacman in pacman_next:
                if abs(next_ghost[0] - next_pacman[0]) + abs(next_ghost[1] - next_pacman[1]) < 2:
                    value = -100_000 - depth
                else:
                    value = self._adversarial_value(next_ghost, next_pacman, depth - 1)
                worst = min(worst, value)
                if worst <= best:
                    break
            best = max(best, worst)
        self.search_cache[key] = best
        return best

    def _visible_minimax_move(self, my_position, pacman_position):
        """Iterative-deepening maximin with a safe deterministic fallback."""
        actions = [(Move.STAY, my_position)] + [
            (move, nxt) for nxt, move in self.neighbors.get(my_position, ())
        ]
        actions.sort(key=lambda item: self._distance(item[1], pacman_position), reverse=True)
        completed_move = actions[0][0]
        self.search_deadline = time.perf_counter() + 0.20

        for depth in range(1, 5):
            self.search_cache = {}
            try:
                depth_best = None
                for move, landing in actions:
                    worst = float("inf")
                    for pac_landing in self._pacman_landings(pacman_position):
                        if abs(landing[0] - pac_landing[0]) + abs(landing[1] - pac_landing[1]) < 2:
                            value = -100_000 - depth
                        else:
                            value = self._adversarial_value(landing, pac_landing, depth - 1)
                        worst = min(worst, value)
                    candidate = (
                        worst,
                        -self.recent_positions.count(landing),
                        int(move != Move.STAY),
                        self.rng.random(),
                    )
                    if depth_best is None or candidate > depth_best[0]:
                        depth_best = candidate, move
                completed_move = depth_best[1]
            except TimeoutError:
                break
        return completed_move

    def _blind_maximin_move(self, my_position):
        """Robust action against every currently plausible Pacman position."""
        belief = self.enemy_belief or self.enemy_start_region or self.open_cells
        # The nearest states determine the worst case. Keeping 80 preserves
        # adversarial behavior while bounding work on larger maps.
        threats = sorted(belief, key=lambda p: self._distance(my_position, p))[:80]
        pacman_next = set()
        for threat in threats:
            pacman_next.update(self._pacman_landings(threat))

        available = [(Move.STAY, my_position)] + [
            (move, nxt) for nxt, move in self.neighbors.get(my_position, ())
        ]
        non_reverse = [item for item in available if item[1] != self.previous_position]
        if len(non_reverse) > 1:
            available = non_reverse

        best = None
        for move, landing in available:
            graph_distances = [self._distance(landing, p) for p in pacman_next]
            capture_count = sum(
                abs(landing[0] - p[0]) + abs(landing[1] - p[1]) < 2
                for p in pacman_next
            )
            exposure_count = sum(self._line_visible(landing, p) for p in pacman_next)
            score = (
                -capture_count,
                min(graph_distances) if graph_distances else 0,
                -exposure_count,
                -self.recent_positions.count(landing),
                -self.visit_count.get(landing, 0),
                self._degree(landing),
                self._local_space(landing),
                self.rng.random(),
            )
            if best is None or score > best[0]:
                best = score, move
        return best[1] if best else Move.STAY

    def _escape_score(self, landing, pacman_position):
        threats = self._pacman_landings(pacman_position)
        distances = [self._distance(landing, p) for p in threats]
        capture_actions = sum(
            abs(landing[0] - p[0]) + abs(landing[1] - p[1]) < 2
            for p in threats
        )
        immediate_loss = capture_actions > 0

        # Prefer breaking the cross-shaped line of sight. This matters beyond
        # raw graph distance: an unseen Ghost denies the seeker an exact target.
        aligned = False
        if landing[0] == pacman_position[0]:
            lo, hi = sorted((landing[1], pacman_position[1]))
            aligned = all((landing[0], c) in self.open_cells for c in range(lo, hi + 1))
        elif landing[1] == pacman_position[1]:
            lo, hi = sorted((landing[0], pacman_position[0]))
            aligned = all((r, landing[1]) in self.open_cells for r in range(lo, hi + 1))

        # Voronoi-like safe territory: cells Ghost reaches strictly before a
        # speed-2 Pacman. Large territory avoids locally safe moves that enter
        # a corridor from which there is no long-term escape.
        ghost_dist = self._distances(landing)
        pac_dist = self._distances(pacman_position)
        safe_area = sum(
            ghost_dist.get(cell, 10_000) < (pac_dist.get(cell, 10_000) + 1) // 2
            for cell in self.open_cells
        )

        # Approximate the next minimax ply: after Pacman's best landing, how
        # many safe replies remain for Ghost?
        closest_threat = min(threats, key=lambda p: self._distance(landing, p))
        replies = {landing}
        replies.update(nxt for nxt, _ in self.neighbors.get(landing, ()))
        safe_replies = sum(
            min(self._distance(reply, p) for p in self._pacman_landings(closest_threat)) >= 2
            for reply in replies
        )
        return (
            -int(immediate_loss),
            -capture_actions,
            -int(aligned),
            min(distances),
            -self.recent_positions.count(landing),
            safe_area,
            safe_replies,
            self._degree(landing),
            self._local_space(landing),
            -self.visit_count.get(landing, 0),
        )

    def _blind_patrol(self, my_position):
        available = list(self.neighbors.get(my_position, ()))
        # Reversing in a corridor produced a short, deterministic two-cell
        # loop that A/B/C could intercept without ever seeing Ghost. Continue
        # forward or turn whenever another legal route exists.
        non_reverse = [
            item for item in available
            if item[0] != self.previous_position
        ]
        if non_reverse:
            available = non_reverse

        choices = []
        for nxt, move in available:
            score = (
                -self.recent_positions.count(nxt),
                -self.visit_count.get(nxt, 0),
                self._degree(nxt),
                self._local_space(nxt),
                self.rng.random(),
            )
            choices.append((score, move))
        if not choices:
            return Move.STAY
        return max(choices, key=lambda item: item[0])[1]

    def step(self, map_state: np.ndarray, my_position: tuple,
             enemy_position: tuple, step_number: int) -> Move:
        self._learn_map(map_state)
        if self.enemy_start_region is None:
            cutoff = map_state.shape[0] * 0.6
            self.enemy_start_region = {p for p in self.open_cells if p[0] > cutoff}
        self.visit_count[my_position] = self.visit_count.get(my_position, 0) + 1
        self.recent_positions.append(my_position)
        self._update_pacman_belief(map_state, enemy_position, step_number)

        if enemy_position is None:
            move = self._blind_maximin_move(my_position)
        else:
            move = self._visible_minimax_move(my_position, enemy_position)

        self.previous_position = my_position
        return move
