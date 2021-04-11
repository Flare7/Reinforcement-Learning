
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future


import matplotlib.pyplot as plt
import numpy as np


NUM_TRIALS = 10000
#EPS = 0.1
BANDIT_PROBABILITIES = [0.2, 0.5, 0.75]


class Bandit:
  def __init__(self, p):
    # p: the win rate
    self.p = p
    self.p_estimate = 5.
    self.N = 1. # num samples collected so far

  def pull(self):
    # draw a 1 with probability p, Bernoulli distribution
    return np.random.random() < self.p

  def update(self, x):
    self.N += 1.
    #formula which was derived in video and i have written it down 
    #If parameter x is True it means 1, False = 0
    self.p_estimate = ((self.N - 1)*self.p_estimate + x) / self.N

def experiment():
  #making instances to class of Bandit
  bandits = [Bandit(p) for p in BANDIT_PROBABILITIES]
  
  #Making list of rewards
  rewards = np.zeros(NUM_TRIALS)
  #finding optimal bandit according to its likelihood
  optimal_j = np.argmax([b.p_estimate for b in bandits])
  print("optimal j:", optimal_j)

  #Iterating though 10K 
  for i in range(NUM_TRIALS):
    j = np.argmax([b.p_estimate for b in bandits])

    # pull the arm for the bandit with the largest sample
    x = bandits[j].pull()
    #print(x)
    # update rewards log
    rewards[i] = x

    # update the distribution for the bandit whose arm we just pulled
    bandits[j].update(x)

    

  # print mean estimates for each bandit
  for b in bandits:
    print("mean estimate:", b.p_estimate)

  # print total reward
  print("total reward earned:", rewards.sum())
  print("overall win rate:", rewards.sum() / NUM_TRIALS)
  print("num times selected each bandit",[b.N for b in bandits])

  # plot the results
  cumulative_rewards = np.cumsum(rewards)
  #print(cumulative_rewards)
  win_rates = cumulative_rewards / (np.arange(NUM_TRIALS) + 1)
  plt.ylim([0,1])
  plt.plot(win_rates)
  plt.plot(np.ones(NUM_TRIALS)*np.max(BANDIT_PROBABILITIES))
  plt.show()

if __name__ == "__main__":
  experiment()
