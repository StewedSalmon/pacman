import sys
import time 
from collections import deque
from pathlib import Path
import random
import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hybrid Phantom v2"
        self._history = []
        self._history_ban = 4  
        self.last_known_enemy_pos = None 
        
        # Tá»‘c Ä‘á»™ giáº£ Ä‘á»‹nh cá»§a Pacman
        self.pacman_speed = max(2, int(kwargs.get("pacman_speed", 2))) 
        
        # Cáº¥u trÃºc báº£n Ä‘á»“
        self.map_analyzed = False
        self.valid_tiles = {}
        self.dead_ends = set()
        self.junctions = set()
        self.map_BFS_cache = {}
        self.dist_to_junction = {} # Má»šI: Báº£n Ä‘á»“ khoáº£ng cÃ¡ch thoÃ¡t hiá»ƒm

    def _analyze_map(self, map_state):
        if self.map_analyzed:
            return
        
        height, width = map_state.shape
        # 1. QuÃ©t map cÆ¡ báº£n
        for r in range(height):
            for c in range(width):
                if map_state[r, c] in [0, -1]:
                    self.valid_tiles[(r, c)] = True
                    if self._cell_exits((r, c), map_state) >= 3:
                        self.junctions.add((r, c))
                        
        # 2. XÃ¢y dá»±ng Ä‘á»“ thá»‹ & NgÃµ cá»¥t
        graph = {pos: [] for pos in self.valid_tiles}
        for pos in self.valid_tiles:
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
                if next_pos in self.valid_tiles:
                    graph[pos].append(next_pos)
                    
        initial_dead_ends = [pos for pos, neighbors in graph.items() if len(neighbors) <= 1]
        queue = deque(initial_dead_ends)
        
        while queue:
            current = queue.popleft()
            self.dead_ends.add(current)
            for neighbor in graph[current]:
                if neighbor not in self.dead_ends:
                    exits = sum(1 for n in graph[neighbor] if n not in self.dead_ends)
                    if exits <= 1:
                        queue.append(neighbor)
                        
        # 3. Má»šI: Multi-source BFS tÃ­nh khoáº£ng cÃ¡ch tá»›i NgÃ£ TÆ° gáº§n nháº¥t
        if self.junctions:
            q_junc = deque(self.junctions)
            for j in self.junctions:
                self.dist_to_junction[j] = 0
            while q_junc:
                curr = q_junc.popleft()
                for m, n_pos in self._get_valid_moves(curr):
                    if n_pos not in self.dist_to_junction:
                        self.dist_to_junction[n_pos] = self.dist_to_junction[curr] + 1
                        q_junc.append(n_pos)
        else:
            for pos in self.valid_tiles:
                self.dist_to_junction[pos] = 0

        self.map_analyzed = True

    def _is_valid_position(self, pos, map_state=None):
        return pos in self.valid_tiles

    def _get_valid_moves(self, pos, map_state=None):
        moves = []
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            next_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if next_pos in self.valid_tiles:
                moves.append((move, next_pos))
        return moves

    def _cell_exits(self, pos, map_state):
        exits = 0
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            n_pos = (pos[0] + move.value[0], pos[1] + move.value[1])
            if n_pos in self.valid_tiles:
                exits += 1
        return exits

    def get_BFS_map(self, start: tuple):
        if start in self.map_BFS_cache:
            return self.map_BFS_cache[start]
        
        dist = {start: 0}
        parent = {start: None}
        q = deque([start])
        
        while q:
            curr = q.popleft()
            for move, next_pos in self._get_valid_moves(curr):
                if next_pos not in dist:
                    dist[next_pos] = dist[curr] + 1
                    parent[next_pos] = (curr, move)
                    q.append(next_pos)
                    
        self.map_BFS_cache[start] = (dist, parent)
        if len(self.map_BFS_cache) > 40:
            self.map_BFS_cache.pop(next(iter(self.map_BFS_cache)))
        return dist, parent

    def is_in_los(self, p1, p2, map_state):
        r1, c1 = p1
        r2, c2 = p2
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1][c] == 1: return False
            return True
        if c1 == c2:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r][c1] == 1: return False
            return True
        return False

    def get_voronoi_area(self, g_pos, p_dist_map):
        g_dist, _ = self.get_BFS_map(g_pos)
        safe_area = 0
        for pos, g_d in g_dist.items():
            p_d = p_dist_map.get(pos, 0)
            if g_d < (p_d / 1.5):
                safe_area += 1
        return safe_area

    def pacman_tick(self, path: list) -> int:
        if not path: return 0
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

    def minimax(self, ghost_pos: tuple, pacman_pos: tuple, map_state: np.ndarray, depth: int, alpha: float, beta: float, ghost_turn: bool):
        if ghost_pos == pacman_pos:
            return -10000 - depth 

        if depth == 0:
            p_dist, p_parent = self.get_BFS_map(pacman_pos)
            
            if ghost_pos not in p_dist:
                return float('inf') 
            
            dist_to_pacman = p_dist[ghost_pos]
            if dist_to_pacman == 0:
                return -10000

            path = []
            curr = ghost_pos
            while curr != pacman_pos:
                prev, move_taken = p_parent[curr]
                path.append(move_taken)
                curr = prev
            path.reverse()
            
            ticks = self.pacman_tick(path)
            score = (ticks * 100) + len(path)
            
            if ghost_pos in self.dead_ends:
                score -= 5000 
                
            score += self.get_voronoi_area(ghost_pos, p_dist) * 10 
            
            if ghost_pos in self.junctions:
                score += 50 
                
            if self.is_in_los(ghost_pos, pacman_pos, map_state):
                score -= 200 
            
            # --- VÃ Lá»–I HORIZON EFFECT: Há»† THá»NG Cáº¢NH BÃO ÄÆ¯á»œNG Háº¦M ---
            d_junc = self.dist_to_junction.get(ghost_pos, 0)
            if d_junc > 0:
                # Náº¿u thá»i gian Pacman Ä‘uá»•i tá»›i nhá» hÆ¡n thá»i gian Ghost láº¿t Ä‘Æ°á»£c ra khá»i háº»m -> Tá»± sÃ¡t!
                if (dist_to_pacman / 1.5) <= d_junc:
                    score -= 3000 
                    
            return score
            
        if ghost_turn:
            max_eval = -float('inf')
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                next_g = (ghost_pos[0] + move.value[0], ghost_pos[1] + move.value[1]) if move != Move.STAY else ghost_pos
                
                if self._is_valid_position(next_g) or move == Move.STAY:
                    eval_score = self.minimax(next_g, pacman_pos, map_state, depth - 1, alpha, beta, False)
                    max_eval = max(max_eval, eval_score)
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break 
            return max_eval
            
        else:
            min_eval = float('inf')
            valid_moves_exist = False
            
            # --- VÃ Lá»–I XUYÃŠN THáº¤U: QUÃ‰T VA CHáº M TRÃŠN Tá»ªNG BÆ¯á»šC CHáº Y ---
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                curr = pacman_pos
                caught = False
                steps_taken = 0
                
                for _ in range(self.pacman_speed):
                    n_pos = (curr[0] + move.value[0], curr[1] + move.value[1])
                    if not self._is_valid_position(n_pos):
                        break
                    curr = n_pos
                    steps_taken += 1
                    
                    # Pacman cháº¡m trÃºng Ghost trong lÃºc Ä‘ang phi nÆ°á»›c Ä‘áº¡i
                    if curr == ghost_pos:
                        caught = True
                        break
                        
                if caught:
                    eval_score = -10000 - depth
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                elif steps_taken > 0:
                    valid_moves_exist = True
                    eval_score = self.minimax(ghost_pos, curr, map_state, depth - 1, alpha, beta, True)
                    min_eval = min(min_eval, eval_score)
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break
                        
            if not valid_moves_exist and min_eval == float('inf'):
                return 0 
            return min_eval

    def step(self, map_state: np.ndarray, my_position: tuple, enemy_position: tuple, step_number: int) -> Move:
        me = tuple(int(v) for v in my_position)
        
        if not self.map_analyzed:
            self._analyze_map(map_state)

        if enemy_position is not None:
            self.last_known_enemy_pos = tuple(int(v) for v in enemy_position)
            
        target_threat = enemy_position or self.last_known_enemy_pos

        if target_threat is None:
            moves = self._get_valid_moves(me)
            if moves:
                return random.choice(moves)[0]
            return Move.STAY

        p_pos = tuple(int(v) for v in target_threat)

        best_move = Move.STAY
        max_eval = -float('inf')
        alpha = -float('inf')
        beta = float('inf')
        
        depth_limit = 3 

        for move, next_pos in self._get_valid_moves(me):
            eval_score = self.minimax(next_pos, p_pos, map_state, depth_limit, alpha, beta, False)
            
            if eval_score > max_eval:
                max_eval = eval_score
                best_move = move
                
            alpha = max(alpha, eval_score)

        return best_move