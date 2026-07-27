import numpy as np
from collections import deque
from environment import Move

def get_manhattan_dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def is_valid(map_state, pos):
    r, c = pos
    rows, cols = map_state.shape
    return 0 <= r < rows and 0 <= c < cols and map_state[r][c] != 1

def update_internal_map(current_map, new_map_state):
    if current_map is None:
        return np.copy(new_map_state)
    
    mask = new_map_state != -1
    current_map[mask] = new_map_state[mask]
    return current_map

def find_nearest_target(map_state, start, target_val=-1):
    """
    Locates the nearest grid cell containing the specified target value using BFS
    """
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

def update_belief_state(belief_map, internal_map, visible_cells, enemy_pos):
    """
    Updates the probability distribution matrix representing the enemy's likely locations
    """
    rows, cols = internal_map.shape
    new_belief = np.zeros((rows, cols))
    
    if enemy_pos:
        new_belief[enemy_pos[0], enemy_pos[1]] = 1.0
        return new_belief
        
    for r in range(rows):
        for c in range(cols):
            if belief_map[r, c] > 0:
                valid_neighbors = []
                for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]:
                    nr, nc = r + move.value[0], c + move.value[1]
                    if 0 <= nr < rows and 0 <= nc < cols and internal_map[nr][nc] != 1:
                        valid_neighbors.append((nr, nc))
                
                if valid_neighbors:
                    prob_share = belief_map[r, c] / len(valid_neighbors)
                    for nr, nc in valid_neighbors:
                        new_belief[nr, nc] += prob_share
                    
    for r, c in visible_cells:
        new_belief[r, c] = 0.0
        
    total_prob = np.sum(new_belief)
    if total_prob > 0:
        new_belief /= total_prob
    else:
        unseen_mask = (internal_map == -1)
        if np.any(unseen_mask):
            new_belief[unseen_mask] = 1.0 / np.sum(unseen_mask)
            
    return new_belief

def calculate_safe_area(internal_map, ghost_pos, pacman_pos):
    rows, cols = internal_map.shape
    
    pacman_dist = np.full((rows, cols), float('inf'))
    pq = deque([(pacman_pos, 0)])
    pacman_dist[pacman_pos] = 0
    
    while pq:
        curr, dist = pq.popleft()
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = curr[0] + move.value[0], curr[1] + move.value[1]
            if 0 <= nr < rows and 0 <= nc < cols and internal_map[nr][nc] != 1:
                if dist + 1 < pacman_dist[nr, nc]:
                    pacman_dist[nr, nc] = dist + 1
                    pq.append(((nr, nc), dist + 1))
                    
    ghost_dist = np.full((rows, cols), float('inf'))
    gq = deque([(ghost_pos, 0)])
    ghost_dist[ghost_pos] = 0
    safe_cells_count = 0
    
    while gq:
        curr, dist = gq.popleft()
        if dist < pacman_dist[curr[0], curr[1]] - 1:
            safe_cells_count += 1
        else:
            continue
            
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nr, nc = curr[0] + move.value[0], curr[1] + move.value[1]
            if 0 <= nr < rows and 0 <= nc < cols and internal_map[nr][nc] != 1:
                if dist + 1 < ghost_dist[nr, nc]:
                    ghost_dist[nr, nc] = dist + 1
                    gq.append(((nr, nc), dist + 1))
                    
    return safe_cells_count

def get_highest_probability_target(belief_map, my_pos):
    """
    Identifies the target node with the highest probability, resolving ties via proximity
    """
    max_prob = np.max(belief_map)
    if max_prob <= 0:
        return None
        
    candidates = np.argwhere(belief_map == max_prob)
    best_target = None
    min_dist = float('inf')
    
    for cand in candidates:
        target = (cand[0], cand[1])
        dist = get_manhattan_dist(my_pos, target)
        if dist < min_dist:
            min_dist = dist
            best_target = target
            
    return best_target