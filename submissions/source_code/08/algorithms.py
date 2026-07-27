import heapq
from collections import deque
from environment import Move
from F import get_manhattan_dist, is_valid

def a_star(map_state, start, goal):
    if not goal or start == goal:
        return None
    
    pq = [(get_manhattan_dist(start, goal), 0, start)]
    g_score = {start: 0}
    first_move = {}
    
    while pq:
        f, g, current = heapq.heappop(pq)
        
        if current == goal:
            return first_move.get(goal, None)
            
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            nxt = (current[0] + move.value[0], current[1] + move.value[1])
            
            if is_valid(map_state, nxt):
                new_g = g_score[current] + 1
                
                if nxt not in g_score or new_g < g_score[nxt]:
                    g_score[nxt] = new_g
                    f_score = new_g + get_manhattan_dist(nxt, goal)
                    heapq.heappush(pq, (f_score, 0, nxt))
                    
                    if current == start:
                        first_move[nxt] = move
                    else:
                        first_move[nxt] = first_move[current]
                        
    return None