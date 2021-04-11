from __future__ import print_function, division
from builtins import range

import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

GAMMA = 0.9
ALPHA = 0.1
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

def random_action(a,eps=0.1):
    num = np.random.random()
    if num < (1-eps):
        return a
    else:
        return np.random.choice(ALL_POSSIBLE_ACTIONS)
    
def play_game(grid,policy):
    s = (2,0)
    grid.set_state(s)
    state_and_rewards = [(s,0)]
    while not grid.game_over(): 
        a = policy[s]
        a = random_action(a)
        r = grid.move(a)
        s = grid.current_state()
        state_and_rewards.append((s,r))
    return state_and_rewards
      
if __name__ == "__main__":
    grid = standard_grid()
    
    # print rewards
    print("rewards:")
    print_values(grid.rewards, grid)
    
    policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'U',
     }
    
    V = {}
    for s in grid.all_states():
        V[s] = 0
        
    for it in range(1000):
        state_and_rewards = play_game(grid,policy)
        for i in range(len(state_and_rewards)-1):
            s, _ = state_and_rewards[i]
            s2, r = state_and_rewards[i+1]
            V[s] = V[s] + ALPHA*(r+GAMMA*V[s2]-V[s])
        
    print("values:")
    print_values(V, grid)
    print("policy:")
    print_policy(policy, grid)
       
        

        
        
        
        
        
        
        
        
        
        
