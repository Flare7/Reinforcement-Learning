from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import windy_grid, ACTION_SPACE
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 1e-3
GAMMA = 0.9

# copied from iterative_policy_evaluation
def get_transition_probs_and_rewards(grid):
  ### define transition probabilities and grid ###
  # the key is (s, a, s'), the value is the probability
  # that is, transition_probs[(s, a, s')] = p(s' | s, a)
  # any key NOT present will considered to be impossible (i.e. probability 0)
  transition_probs = {}

  # to reduce the dimensionality of the dictionary, we'll use deterministic
  # rewards, r(s, a, s')
  # note: you could make it simpler by using r(s') since the reward doesn't
  # actually depend on (s, a)
  rewards = {}

  for (s, a), v in grid.probs.items():
    print(s,a,v)
    for s2, p in v.items():
      transition_probs[(s, a, s2)] = p
      rewards[(s, a, s2)] = grid.rewards.get(s2, 0)

  return transition_probs, rewards

if __name__ == '__main__':
  grid = windy_grid()
  transition_probs, rewards = get_transition_probs_and_rewards(grid)
  
  #In this, we are not stating policy, first of all we are going to evaluate each state of grid with V(s)
  #Then we are going to take another loop to state the action 
  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # initialize V(s)
  V = {}
  states = grid.all_states()
  for s in states:
    V[s] = 0

  # repeat until convergence
  # V[s] = max[a]{ sum[s',r] { p(s',r|s,a)[r + gamma*V[s']] } }
  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = float('-inf')

        for a in ACTION_SPACE:
          v = 0
          for s2 in grid.all_states():
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

          # keep v if it's better
          if v > new_v:
            new_v = v

        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))

    it += 1
    if biggest_change < SMALL_ENOUGH:
      break

  # find a policy that leads to optimal value function
  policy = {}
  for s in grid.actions.keys():
    best_a = None
    best_value = float('-inf')
    # loop through all possible actions to find the best current action
    for a in ACTION_SPACE:
      v = 0
      for s2 in grid.all_states():
        #(0, 1),(1, 2),(0, 0),(1, 3),(2, 1),(2, 0),(2, 3),(2, 2),(1, 0),(0, 2),(0, 3) - S2
        # reward is a function of (s, a, s'), 0 if not specified
        r = rewards.get((s, a, s2), 0)
        v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

      # best_a is the action associated with best_value
      if v > best_value:
        best_value = v
        best_a = a
      print()
    policy[s] = best_a

  # our goal here is to verify that we get the same answer as with policy iteration
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)