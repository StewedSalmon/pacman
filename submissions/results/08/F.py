import numpy as np
import heapq
from collections import deque
from environment import Move

def get_manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def is_valid(map_state, pos):
    r, c = pos
    rows, cols = map_state.shape
    return 0 <= r < rows and 0 <= c < cols and map_state[r][c] != 1

# Cập nhật bản đồ nội bộ của Agent dựa trên tầm nhìn mới
def update_internal_map(current_map, new_map_state):
    if current_map is None:
        return np.copy(new_map_state)
    
    mask = new_map_state != -1
    current_map[mask] = new_map_state[mask]
    return current_map


def find_nearest_target(map_state, start, target_val=-1):
    #BFS tìm tọa độ gần nhất có giá trị target_val
    queue = deque([start])
    visited = {start}
    
    while queue:
        current = queue.popleft()
        if map_state[current[0]][current[1]] == target_val:
            return current
            
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (current[0] + move.value[0], current[1] + move.value[1])
            if is_valid(map_state, nxt) and nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)
    return None

