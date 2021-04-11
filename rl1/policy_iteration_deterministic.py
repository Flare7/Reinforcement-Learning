
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import numpy as np
from grid_world import standard_grid, ACTION_SPACE
from iterative_policy_evaluation_deterministic import print_values, print_policy

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

  for i in range(grid.rows):
    for j in range(grid.cols):
      s = (i, j)
      if not grid.is_terminal(s):
        for a in ACTION_SPACE:
          #This function returns all possible states in grid which can be achieved
          s2 = grid.get_next_state(s, a)
          #And then making a probability to step on that tile as 1
          transition_probs[(s, a, s2)] = 1
          #If that state s2 is in rewards dic then go on
          if s2 in grid.rewards:
            rewards[(s, a, s2)] = grid.rewards[s2]

  return transition_probs, rewards

#Similar code as previous 
def evaluate_deterministic_policy(grid, policy):
  # initialize V(s) = 0
  V = {}
  for s in grid.all_states():
    V[s] = 0

  # repeat until convergence
  it = 0
  while True:
    biggest_change = 0
    for s in grid.all_states():
      if not grid.is_terminal(s):
        old_v = V[s]
        new_v = 0 # we will accumulate the answer
        for a in ACTION_SPACE:
          for s2 in grid.all_states():

            # action probability is deterministic
            action_prob = 1 if policy.get(s) == a else 0
            
            # reward is a function of (s, a, s'), 0 if not specified
            r = rewards.get((s, a, s2), 0)
            new_v += action_prob * transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

        # after done getting the new value, update the value table
        V[s] = new_v
        biggest_change = max(biggest_change, np.abs(old_v - V[s]))
    it += 1

    if biggest_change < SMALL_ENOUGH:
      break
  return V


if __name__ == '__main__':

  grid = standard_grid()
  transition_probs, rewards = get_transition_probs_and_rewards(grid)

  # print rewards
  print("rewards:")
  print_values(grid.rewards, grid)

  # state -> action
  # we'll randomly choose an action and update as we learn
  #{(0, 0): 'U', (0, 1): 'D', (0, 2): 'R', (1, 0): 'L', (1, 2): 'D', (2, 0): 'U', (2, 1): 'D', (2, 2): 'L', (2, 3): 'L'}
  #randomly pick an action and it will be updated after the algo
  policy = {}
  for s in grid.actions.keys():
    policy[s] = np.random.choice(ACTION_SPACE)
  
  # initial policy
  print("initial policy:")
  print_policy(policy, grid)

  # repeat until convergence - will break out when policy does not change
  #This loop will iterate through each tile and find the best action according to V(s) fucntion and change the action after investigation
  #The action will be found according to iteration of 4 possible actions and the highest Q(a) value will be displayed into police[s] 
  #To check it, look up to depiction of grid world 
  while True:

    # policy evaluation step - we already know how to do this!
    #evaluate each tiles of grid world
    V = evaluate_deterministic_policy(grid, policy)

    # policy improvement step
    is_policy_converged = True
    for s in grid.actions.keys():
      old_a = policy[s]
      new_a = None
      best_value = float('-inf')

      # loop through all possible actions to find the best current action,
      # This loop will find the best action depends on value function 
      for a in ACTION_SPACE:
        v = 0
        for s2 in grid.all_states():
          #print(s2)
          # reward is a function of (s, a, s'), 0 if not specified
          r = rewards.get((s, a, s2), 0)
          v += transition_probs.get((s, a, s2), 0) * (r + GAMMA * V[s2])

        if v > best_value:
          best_value = v
          new_a = a

      # new_a now represents the best action in this state
      #If the actions are not equal to each other than it will entry the loop 
      policy[s] = new_a
      if new_a != old_a:
        is_policy_converged = False

    if is_policy_converged:
      break

  # once we're done, print the final policy and values
  print("values:")
  print_values(V, grid)
  print("policy:")
  print_policy(policy, grid)