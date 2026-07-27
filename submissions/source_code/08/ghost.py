import numpy as np
from collections import deque
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent

class CustomPRNG:
    def __init__(self, seed=999):
        self.state = seed
    def random(self):
        self.state = (1103515245 * self.state + 12345) & 0x7fffffff
        return self.state / 0x7fffffff
    def choice(self, seq):
        if not seq: return None
        return seq[int(self.random() * len(seq))]

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self.internal_map = None
        self.last_enemy_pos = None
        self.prev_pos = None
        self.prng = CustomPRNG(seed=123)

    def update_memory(self, map_state, enemy_pos):
        if self.internal_map is None:
            self.internal_map = np.copy(map_state)
        else:
            mask = map_state != -1
            self.internal_map[mask] = map_state[mask]
            
        if enemy_pos:
            self.last_enemy_pos = enemy_pos

    def get_valid_moves(self, pos, forbid_pos=None):
        moves = []
        rows, cols = self.internal_map.shape
        for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = pos[0] + m.value[0], pos[1] + m.value[1]
            if 0 <= nr < rows and 0 <= nc < cols and self.internal_map[nr, nc] != 1:
                moves.append((m, (nr, nc)))
                
        if forbid_pos and len(moves) > 1:
            moves = [m for m in moves if m[1] != forbid_pos]
            
        return moves

    def get_all_distances(self, start):
        rows, cols = self.internal_map.shape
        dist_map = np.full((rows, cols), 9999)
        if not start: return dist_map
        
        queue = deque([(start, 0)])
        dist_map[start[0], start[1]] = 0
        
        while queue:
            curr, d = queue.popleft()
            for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                nr, nc = curr[0] + m.value[0], curr[1] + m.value[1]
                if 0 <= nr < rows and 0 <= nc < cols and self.internal_map[nr, nc] != 1:
                    if dist_map[nr, nc] == 9999:
                        dist_map[nr, nc] = d + 1
                        queue.append(((nr, nc), d + 1))
        return dist_map

    def evaluate_voronoi(self, ghost_next_pos, pacman_pos):
        import math
        ghost_dists = self.get_all_distances(ghost_next_pos)
        pacman_dists = self.get_all_distances(pacman_pos)
        
        safe_score = 0.0
        rows, cols = self.internal_map.shape
        for r in range(rows):
            for c in range(cols):
                if self.internal_map[r, c] != 1:
                    pacman_turns = math.ceil(pacman_dists[r, c] / self.pacman_speed)
                    ghost_turns = ghost_dists[r, c]
                    
                    if ghost_turns < pacman_turns:
                        exits = sum(1 for m in [(0,1), (0,-1), (1,0), (-1,0)] 
                                    if 0 <= r+m[0] < rows and 0 <= c+m[1] < cols 
                                    and self.internal_map[r+m[0], c+m[1]] != 1)
                        
                        if exits <= 1:
                            safe_score += 0.1   
                        elif exits == 2:
                            safe_score += 1.0  
                        else:
                            safe_score += 3.0 
                        
        dist_to_pacman = pacman_dists[ghost_next_pos[0], ghost_next_pos[1]]
        return safe_score, dist_to_pacman

    def step(self, map_state, my_pos, enemy_pos, step_number):
        self.update_memory(map_state, enemy_pos)

        if step_number <= 10:
            rows, cols = self.internal_map.shape
            valid_dirs = []
            for m in [Move.DOWN, Move.RIGHT]:
                nr, nc = my_pos[0] + m.value[0], my_pos[1] + m.value[1]
                if 0 <= nr < rows and 0 <= nc < cols and self.internal_map[nr, nc] != 1:
                    valid_dirs.append(m)
            
            best_move = Move.STAY
            if Move.DOWN in valid_dirs:
                best_move = Move.DOWN
            elif Move.RIGHT in valid_dirs:
                best_move = Move.RIGHT
            elif valid_dirs:
                best_move = self.prng.choice(valid_dirs)
                
            if best_move != Move.STAY:
                self.prev_pos = my_pos
            return best_move


        # VORONOI SURVIVAL 
        moves = self.get_valid_moves(my_pos, forbid_pos=self.prev_pos)

        # Nếu không thấy Pacman, đi bọc biên nhưng dứt khoát tiến lên
        if not self.last_enemy_pos:
            best_move = self.prng.choice(moves)[0] if moves else Move.STAY
            if best_move != Move.STAY:
                self.prev_pos = my_pos
            return best_move

        # Nếu có dấu vết Pacman, tính toán vùng an toàn
        best_move = Move.STAY
        best_score = -float('inf')

        for move, nxt_pos in moves:
            safe_area, dist_to_pacman = self.evaluate_voronoi(nxt_pos, self.last_enemy_pos)
            
            if dist_to_pacman <= 1:
                score = -999999
            else:
                score = (safe_area * 100) + dist_to_pacman
                
            if score > best_score:
                best_score = score
                best_move = move
                
        # Nếu bị dồn vào đường cùng cố chọn đường xa nhất
        if best_move == Move.STAY and moves:
            pacman_dists = self.get_all_distances(self.last_enemy_pos)
            best_move = max(moves, key=lambda m: pacman_dists[m[1][0], m[1][1]])[0]

        if best_move != Move.STAY:
            self.prev_pos = my_pos
            
        return best_move