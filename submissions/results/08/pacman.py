import numpy as np
import random
from environment import Move
from agent_interface import PacmanAgent as BasePacmanAgent
from F import get_manhattan_dist, is_valid, update_internal_map, find_nearest_target
from algorithms import a_star

class PacmanAgent(BasePacmanAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = kwargs.get('pacman_speed', 2)
        self.internal_map = None
        self.last_enemy_pos = None
        self.patrol_target = None

    def step(self, map_state, my_pos, enemy_pos, step_number):
        self.internal_map = update_internal_map(self.internal_map, map_state)
        target_move = Move.STAY

        if enemy_pos:
            self.last_enemy_pos = enemy_pos
            
        if self.last_enemy_pos and my_pos == self.last_enemy_pos and not enemy_pos:
            self.last_enemy_pos = None

        # Ưu tiên 1: Đuổi theo vị trí đã biết của địch
        if self.last_enemy_pos:
            target_move = a_star(self.internal_map, my_pos, self.last_enemy_pos)
        
        # Ưu tiên 2: Tìm vùng sương mù (-1) gần nhất để mở map
        if not target_move or target_move == Move.STAY:
            explore_target = find_nearest_target(self.internal_map, my_pos, target_val=-1)
            if explore_target:
                target_move = a_star(self.internal_map, my_pos, explore_target)
                self.patrol_target = None

        # Ưu tiên 3: Đi tuần khi map đã mở hết
        if not target_move or target_move == Move.STAY:
            if (self.patrol_target is None or 
                get_manhattan_dist(my_pos, self.patrol_target) <= self.pacman_speed):
                valid_cells = np.argwhere((self.internal_map == 0) | 
                                          (self.internal_map == 2) | 
                                          (self.internal_map == 3))
                if len(valid_cells) > 0:
                    self.patrol_target = tuple(random.choice(valid_cells))
            
            if self.patrol_target:
                target_move = a_star(self.internal_map, my_pos, self.patrol_target)

        if not target_move or target_move == Move.STAY:
            valid_moves = [m for m in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT] 
                           if is_valid(self.internal_map, (my_pos[0]+m.value[0], my_pos[1]+m.value[1]))]
            target_move = random.choice(valid_moves) if valid_moves else Move.STAY


        steps = 1
        if target_move != Move.STAY:
            if enemy_pos is None and -1 not in self.internal_map:
                steps = 1
            else:
                path_clear = True
                for s in range(1, self.pacman_speed + 1):
                    check_pos = (my_pos[0] + target_move.value[0] * s, 
                                 my_pos[1] + target_move.value[1] * s)
                    if not is_valid(self.internal_map, check_pos):
                        path_clear = False
                        break
                if path_clear:
                    steps = self.pacman_speed

        return (target_move, steps)