from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

# NOTE: this is only policy evaluation, not optimization

def play_game(grid, policy,start_states):
  # returns a list of states and corresponding returns

  # reset game to start at a random position, every time when we want to play game we are going to generate new pos
  # we need to do this, because given our current deterministic policy
  # we would never end up at certain states, but we still want to measure their value
  #start_states = list(grid.actions.keys())
  start_idx = np.random.choice(len(start_states))
  #to inicialize current pos in grid 
  grid.set_state(start_states[start_idx])
  
  #obtain current state/pos
  s = grid.current_state()
  
  #first reward has to be 0 because of definition
  states_and_rewards = [(s, 0)] # list of tuples of (state, reward)
  while not grid.game_over():
    a = policy[s]
    r = grid.move(a)
    s = grid.current_state()
    states_and_rewards.append((s, r))
  
  #states_and_rewards - [((2, 1), 0), ((2, 2), 0), ((1, 2), 0), ((0, 2), 0), ((0, 3), 1)]
  #the algo beneath will calculate the reward for each state above, compute it by previous reward of state (S)
  #result after execution - [((2, 1), 0.7290000000000001), ((2, 2), 0.81), ((1, 2), 0.9), ((0, 2), 1.0)]
  # calculate the returns by working backwards from the terminal state
  G = 0
  states_and_returns = []
  first = True
  for s, r in reversed(states_and_rewards):
    # the value of the terminal state is 0 by definition
    # we should ignore the first state we encounter
    # and ignore the last G, which is meaningless since it doesn't correspond to any move
    if first:
      first = False
    else:
      states_and_returns.append((s, G))
    G = r + GAMMA*G
  states_and_returns.reverse() # we want it to be in order of state visited
  return states_and_returns


if __name__ == '__main__':
  # use the standard grid again (0 for every step) so that we can compare
  # to iterative policy evaluation
  grid = standard_grid()

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  policy = {
    (2, 0): 'U',
    (1, 0): 'U',
    (0, 0): 'R',
    (0, 1): 'R',
    (0, 2): 'R',
    (1, 2): 'U',
    (2, 1): 'R',
    (2, 2): 'U',
    (2, 3): 'L',
  }

  # initialize V(s) and returns
  V = {}
  returns = {} # dictionary of state -> list of returns we've received
  states = grid.all_states()
  for s in states:
    if s in grid.actions:
      #make a list to store rewards for particular states
      returns[s] = [] 
    else:
      # terminal state or state we can't otherwise get to
      V[s] = 0
  
  start_states = list(grid.actions.keys())
  # repeat
  #50 is enough to execute this script 
  for t in range(20):

    # generate an episode using pi , going to play episode 
    states_and_returns = play_game(grid, policy,start_states)
    seen_states = set()
    for s, G in states_and_returns:
      # check if we have already seen s
      # called "first-visit" MC policy evaluation
      if s not in seen_states:
        returns[s].append(G)
        V[s] = np.mean(returns[s])
        seen_states.add(s)

  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)


