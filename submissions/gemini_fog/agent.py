import numpy as np
import random
from collections import deque
from enum import Enum
from pathlib import Path
import sys

# Interface imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


def get_manhattan_distance(pos1: tuple, pos2: tuple) -> int:
    """Calculates Manhattan distance between two coordinates."""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


class GhostState(Enum):
    TRANSIT = 1
    ARRIVED = 2


class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Topology Evasion Ghost"

        self.height = kwargs.get("map_height", 21)
        self.width = kwargs.get("map_width", 21)
        self.pacman_speed = kwargs.get("pacman_speed", 2)
        self.max_depth = 2

        self.explored_map = np.full((self.height, self.width), -1)
        self.heatmap = np.zeros((self.height, self.width))

        self.dead_ends = set()
        self.dynamic_waypoints = []
        self.visited_waypoints = []
        self.target_node = None

        self.state = GhostState.TRANSIT
        self.last_known_enemy_pos = None
        self.is_hunting_blind = False
        self.previous_step = Move.STAY
        self.topology_analyzed = False
        self.map_BFS_cache = {}

    def _analyze_topology_on_step_1(self, map_state: np.ndarray, ghost_spawn: tuple):
        """
        Runs on Step 1.
        Extracts dead ends, identifies the Right Corner for the early slip,
        and targets the Bottom-Half Island for continuous endgame looping.
        """
        non_walls = set()
        for r in range(self.height):
            for c in range(self.width):
                if map_state[r, c] != 1:
                    non_walls.add((r, c))

        # Build adjacency favoring RIGHT / DOWN steps for micro tie-breakers
        adj = {}
        for r, c in non_walls:
            neighbors = []
            for dr, dc in [(0, 1), (1, 0), (-1, 0), (0, -1)]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in non_walls:
                    neighbors.append((nr, nc))
            adj[(r, c)] = neighbors

        # Iterative Dead-End Pruning
        dead_ends = set()
        deg = {node: len(nbrs) for node, nbrs in adj.items()}
        queue = [node for node, d in deg.items() if d == 1]

        while queue:
            curr = queue.pop(0)
            dead_ends.add(curr)
            for nbr in adj[curr]:
                if nbr not in dead_ends:
                    deg[nbr] -= 1
                    if deg[nbr] == 1:
                        queue.append(nbr)

        self.dead_ends = dead_ends
        safe_nodes = non_walls - dead_ends

        # Identify junctions (nodes with >= 3 safe exits)
        junctions = [
            node for node in safe_nodes 
            if len([n for n in adj[node] if n not in dead_ends]) >= 3
        ]

        # BFS from Ghost Spawn across non-wall tiles to calculate distances
        distances = {ghost_spawn: 0}
        bfs_queue = deque([ghost_spawn])
        while bfs_queue:
            curr = bfs_queue.popleft()
            for nbr in adj[curr]:
                if nbr not in distances:
                    distances[nbr] = distances[curr] + 1
                    bfs_queue.append(nbr)

        # 1. WAYPOINT 0: Right Corner (Down-Right Slip to dodge center rush)
        right_corner_candidates = sorted(
            junctions if junctions else list(safe_nodes),
            key=lambda j: distances.get(j, 0) + (j[0] * 4) + (j[1] * 4),
            reverse=True
        )
        right_corner = right_corner_candidates[0] if right_corner_candidates else ghost_spawn

        # 2. WAYPOINTS 1-3: Pacman Base Island Hubs (Bottom-Center)
        bottom_junctions = [j for j in junctions if j[0] > self.height // 2]
        center_c = self.width // 2
        
        if len(bottom_junctions) >= 3:
            # Sort bottom junctions to heavily favor the CENTER column (locking onto the base island)
            bottom_junctions.sort(key=lambda j: abs(j[1] - center_c))
            island_waypoints = bottom_junctions[:3]
        elif bottom_junctions:
            island_waypoints = bottom_junctions
        else:
            # Fallback for simple maps
            bottom_safe = [n for n in safe_nodes if n[0] > self.height // 2]
            bottom_safe.sort(key=lambda j: abs(j[1] - center_c))
            island_waypoints = bottom_safe[:3] if bottom_safe else [ghost_spawn]

        # Assemble explicit sequence: [1. Right Corner Slip] -> [2-4. Base Island Orbit]
        self.dynamic_waypoints = [right_corner] + [w for w in island_waypoints if w != right_corner]
        self.topology_analyzed = True

        # Assemble explicit sequence: [1. Right Corner Slip] -> [2-4. Bottom Half Island Orbit]
        self.dynamic_waypoints = [right_corner] + [w for w in island_waypoints if w != right_corner]
        self.topology_analyzed = True

    def update_explored_map(self, map_state: np.ndarray):
        """Update internal map state from global array."""
        for r in range(self.height):
            for c in range(self.width):
                if map_state[r, c] != -1:
                    if map_state[r, c] in (2, 3):  # 2: Pacman, 3: Ghost
                        self.explored_map[r, c] = 0
                    else:
                        self.explored_map[r, c] = map_state[r, c]

    def is_tile_valid(self, pos: tuple) -> bool:
        """Checks bounds and wall status."""
        r, c = pos
        if 0 <= r < self.height and 0 <= c < self.width:
            return self.explored_map[r, c] == 0
        return False

    def is_tile_safe(self, pos: tuple) -> bool:
        """Checks valid floor tile AND excludes dead ends."""
        return self.is_tile_valid(pos) and (pos not in self.dead_ends)

    def get_BFS_map(self, start: tuple, safe_only: bool = True):
        """Calculates or fetches cached BFS tree from start position."""
        cache_key = (start, safe_only)
        if cache_key in self.map_BFS_cache:
            return self.map_BFS_cache[cache_key]

        distances = {start: 0}
        parent = {start: None}
        queue = deque([start])

        while queue:
            current = queue.popleft()
            for move in [Move.DOWN, Move.RIGHT, Move.UP, Move.LEFT]:
                nr, nc = current[0] + move.value[0], current[1] + move.value[1]
                next_pos = (nr, nc)

                valid = self.is_tile_safe(next_pos) if safe_only else self.is_tile_valid(next_pos)
                if valid and next_pos not in distances:
                    distances[next_pos] = distances[current] + 1
                    parent[next_pos] = (current, move)
                    queue.append(next_pos)

        self.map_BFS_cache[cache_key] = (distances, parent)
        return distances, parent

    def get_visible_cells(self, pos: tuple) -> set:
        """Gets all line-of-sight cells in straight rays until a wall."""
        visible = {pos}
        for move in [Move.RIGHT, Move.DOWN, Move.UP, Move.LEFT]:
            dr, dc = move.value
            r, c = pos
            for _ in range(self.width):
                r += dr
                c += dc
                if not (0 <= r < self.height and 0 <= c < self.width):
                    break
                if self.explored_map[r, c] == 1:
                    break
                visible.add((r, c))
        return visible

    def pick_next_waypoint(self, my_position: tuple) -> tuple:
        """
        Sequential Waypoint Selection:
        1. Moves to Right Corner first.
        2. Routes to Bottom-Half Island nodes.
        3. Once arrived at the island, loops strictly among the bottom-half island nodes forever.
        """
        candidates = [w for w in self.dynamic_waypoints if w not in self.visited_waypoints and w != my_position]
        
        # When all waypoints have been visited, reset visited_waypoints to KEEP the Right Corner marked as visited.
        # This prevents the Ghost from ever travelling back to the Right Corner, locking it into the Bottom Island loop.
        if not candidates:
            if len(self.dynamic_waypoints) > 1:
                self.visited_waypoints = [self.dynamic_waypoints[0]]  # Lock out Right Corner (index 0)
            else:
                self.visited_waypoints.clear()
            candidates = [w for w in self.dynamic_waypoints if w not in self.visited_waypoints and w != my_position]

        if not candidates:
            return my_position

        # Maintain strict sequence order (candidates[0]) rather than max-distance reordering
        return candidates[0]

    def get_safe_transit_move(self, current_pos: tuple, target_pos: tuple) -> Move:
        """Finds the best safe move towards target using BFS pathing."""
        distances, parent = self.get_BFS_map(current_pos, safe_only=True)

        if target_pos in distances and target_pos != current_pos:
            # Reconstruct first step from current_pos towards target_pos
            curr = target_pos
            while parent[curr][0] != current_pos:
                curr = parent[curr][0]
            return parent[curr][1]

        # Greedy fallback if BFS path to exact tile isn't found
        best_move = Move.STAY
        min_dist = float('inf')

        for move in [Move.DOWN, Move.RIGHT, Move.UP, Move.LEFT]:
            next_pos = (current_pos[0] + move.value[0], current_pos[1] + move.value[1])
            if self.is_tile_safe(next_pos):
                dist = get_manhattan_distance(next_pos, target_pos)
                if move == self._get_reverse_move(self.previous_step):
                    dist += 3  # Penalty to avoid 180-degree jittering

                if dist < min_dist:
                    min_dist = dist
                    best_move = move

        return best_move

    def _get_reverse_move(self, move: Move) -> Move:
        if move == Move.STAY:
            return Move.STAY
        reverse = {
            Move.UP: Move.DOWN,
            Move.DOWN: Move.UP,
            Move.LEFT: Move.RIGHT,
            Move.RIGHT: Move.LEFT,
        }
        return reverse.get(move, Move.STAY)

    def minimax(self, ghost_pos: tuple, pacman_pos: tuple, depth: int, alpha: float, beta: float, ghost_turn: bool) -> float:
        """Evasion Minimax evaluating survival depth and distance to Pacman."""
        dist = get_manhattan_distance(ghost_pos, pacman_pos)
        if dist <= 1:
            return -1000.0 - depth

        if depth == 0:
            distances, _ = self.get_BFS_map(pacman_pos, safe_only=False)
            path_len = distances.get(ghost_pos, dist)
            visible_count = len(self.get_visible_cells(ghost_pos))
            # Higher score = safer for Ghost (longer path, less visible exposure)
            return (path_len * 10.0) - visible_count

        if ghost_turn:
            max_eval = -float('inf')
            next_moves = []
            for move in [Move.STAY, Move.RIGHT, Move.DOWN, Move.UP, Move.LEFT]:
                next_pos = (ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1])
                if self.is_tile_valid(next_pos):
                    next_moves.append(next_pos)

            # Prioritize positions furthest from Pacman
            next_moves.sort(key=lambda p: get_manhattan_distance(p, pacman_pos), reverse=True)

            for next_pos in next_moves:
                eval_score = self.minimax(next_pos, pacman_pos, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            # Project Pacman 1-step moves
            pacman_moves = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                np_pos = (pacman_pos[0] + move.value[0], pacman_pos[1] + move.value[1])
                if self.is_tile_valid(np_pos):
                    pacman_moves.append(np_pos)

            pacman_moves.sort(key=lambda p: get_manhattan_distance(p, ghost_pos))

            for np_pos in pacman_moves:
                eval_score = self.minimax(ghost_pos, np_pos, depth - 1, alpha, beta, True)
                min_eval = min(min_eval, eval_score)
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            return min_eval

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        self.map_BFS_cache.clear()
        self.update_explored_map(map_state)

        # Step 1: Initialize global maze topology & dead ends
        if not self.topology_analyzed:
            self._analyze_topology_on_step_1(map_state, my_position)

        # Active Line of Sight Pursuit / Evasion
        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            best_move = Move.STAY
            max_eval = -float('inf')
            alpha = -float('inf')
            beta = float('inf')

            # Evaluate moves favoring safe non-dead-end tiles if possible
            valid_moves = []
            for move in [Move.STAY, Move.RIGHT, Move.DOWN, Move.UP, Move.LEFT]:
                next_pos = (my_position[0] + move.value[0], my_position[1] + move.value[1])
                if self.is_tile_valid(next_pos):
                    # Penalize dead-end tiles in score unless forced
                    penalty = -500 if next_pos in self.dead_ends else 0
                    valid_moves.append((move, next_pos, penalty))

            for move, next_pos, penalty in valid_moves:
                eval_score = self.minimax(next_pos, enemy_position, self.max_depth, alpha, beta, False) + penalty
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)

            self.previous_step = best_move
            return best_move

        # Unseen / Transit Exploration Phase
        if self.target_node is None or my_position == self.target_node:
            if self.target_node == my_position:
                self.visited_waypoints.append(self.target_node)
            self.target_node = self.pick_next_waypoint(my_position)

        action = self.get_safe_transit_move(my_position, self.target_node)

        # If trapped or routing fails, pick new waypoint
        if action == Move.STAY and self.target_node != my_position:
            self.visited_waypoints.append(self.target_node)
            self.target_node = self.pick_next_waypoint(my_position)
            action = self.get_safe_transit_move(my_position, self.target_node)

        self.previous_step = action
        return action