import random
from environment import Move
from agent_interface import GhostAgent as BaseGhostAgent
from F import get_manhattan_dist, is_valid, update_internal_map
from algorithms import a_star

# Ghost đơn giản, xoay 
class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.internal_map = None
        self.patrol_nodes = []
        self.current_patrol_idx = 0
        self.current_patrol_target = None
        self.patrol_direction = 1  # 1: Thuận kim đồng hồ, -1: Ngược kim đồng hồ
        self.checkpoint_3h = None
        self.left_checkpoint = False  
        self.is_first_3h_visit = True  # Đánh dấu lần đầu tới mốc 3H

    def _setup_patrol_nodes(self):
        #Tính toán 6 điểm để tuần tra sát biên
        rows, cols = self.internal_map.shape
        valid_cells = [(r, c) for r in range(rows) for c in range(cols) 
                       if self.internal_map[r][c] != 1]
        if not valid_cells:
            return []
        
        tl = min(valid_cells, key=lambda p: p[0] + p[1])
        tr = min(valid_cells, key=lambda p: p[0] - p[1])
        br = max(valid_cells, key=lambda p: p[0] + p[1])
        bl = max(valid_cells, key=lambda p: p[0] - p[1])
        
        mid_row = rows // 2
        valid_mid_row = [p for p in valid_cells if p[0] == mid_row]
        
        if valid_mid_row:
            p_3h = max(valid_mid_row, key=lambda p: p[1]) # Biên phải
            p_9h = min(valid_mid_row, key=lambda p: p[1]) # Biên trái
        else:
            p_3h, p_9h = tr, tl
            
        self.checkpoint_3h = p_3h 
        
        return [tl, tr, p_3h, br, bl, p_9h]

    def _get_evasion_move(self, my_pos, enemy_pos, step_number):
        best_move = Move.STAY
        max_dist = -1
        
        for m in [Move.DOWN, Move.RIGHT, Move.LEFT, Move.UP]:
            if step_number <= 10 and m in [Move.LEFT, Move.UP]:
                continue

            nxt = (my_pos[0] + m.value[0], my_pos[1] + m.value[1])
            if is_valid(self.internal_map, nxt):
                dist = get_manhattan_dist(nxt, enemy_pos)
                if dist > max_dist:
                    max_dist = dist
                    best_move = m
                    
        if best_move == Move.STAY and step_number <= 10:
            for fallback_m in [Move.RIGHT, Move.DOWN, Move.LEFT, Move.UP]:
                nxt = (my_pos[0] + fallback_m.value[0], my_pos[1] + fallback_m.value[1])
                if is_valid(self.internal_map, nxt):
                    return fallback_m
                
        return best_move

    def step(self, map_state, my_pos, enemy_pos, step_number):
        self.internal_map = update_internal_map(self.internal_map, map_state)

        # Khởi tạo mốc ở lượt đầu tiên
        if not self.patrol_nodes:
            self.patrol_nodes = self._setup_patrol_nodes()
            if self.patrol_nodes:
                self.current_patrol_idx = 2  # Nhắm thẳng ra 3H trước
                self.current_patrol_target = self.patrol_nodes[self.current_patrol_idx]

        # Khoảng cách > 5 mới được kích hoạt lại khả năng đảo chiều ở mốc 3H
        if self.checkpoint_3h and get_manhattan_dist(my_pos, self.checkpoint_3h) > 5:
            self.left_checkpoint = True

        move = Move.STAY


        if enemy_pos:
            move = self._get_evasion_move(my_pos, enemy_pos, step_number)
        else:
            reached_target = False
            if self.current_patrol_target and get_manhattan_dist(my_pos, self.current_patrol_target) <= 2:
                reached_target = True

            # Chỉ đảo chiều từ vòng thứ 2 trở đi
            if reached_target and self.current_patrol_target == self.checkpoint_3h:
                if self.is_first_3h_visit:
                    self.is_first_3h_visit = False  # Xóa cờ lần đầu
                    self.left_checkpoint = False    # Khóa lại an toàn
                elif self.left_checkpoint:
                    self.patrol_direction *= -1  
                    self.left_checkpoint = False 

            # Cập nhật mục tiêu tiếp theo khi lướt qua mốc hiện tại
            if reached_target:
                self.current_patrol_idx = (self.current_patrol_idx + self.patrol_direction) % len(self.patrol_nodes)
                self.current_patrol_target = self.patrol_nodes[self.current_patrol_idx]
                
            if self.current_patrol_target:
                move = a_star(self.internal_map, my_pos, self.current_patrol_target)
            
            if not move or move == Move.STAY:
                valid_moves = [m for m in [Move.DOWN, Move.RIGHT, Move.LEFT, Move.UP] 
                              if is_valid(self.internal_map, (my_pos[0]+m.value[0], my_pos[1]+m.value[1]))]
                move = random.choice(valid_moves) if valid_moves else Move.STAY

        # Khoá 10 steps đầu
        if step_number <= 10 and move in [Move.LEFT, Move.UP]:
            valid_moves = [m for m in [Move.DOWN, Move.RIGHT] 
                          if is_valid(self.internal_map, (my_pos[0]+m.value[0], my_pos[1]+m.value[1]))]
            if valid_moves:
                move = random.choice(valid_moves)

        return move