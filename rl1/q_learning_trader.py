

#!wget -nc https://lazyprogrammer.me/course_files/sp500_closefull.csv

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os

print(os.getcwd())

df0 = pd.read_csv('sp500_closefull.csv', index_col=0, parse_dates=True)
df0.dropna(axis=0, how='all', inplace=True)
df0.dropna(axis=1, how='any', inplace=True)

df_returns = pd.DataFrame()
for name in df0.columns:
  df_returns[name] = np.log(df0[name]).diff()

# split into train and test
Ntest = 1000
train_data = df_returns.iloc[:-Ntest]
test_data = df_returns.iloc[-Ntest:]
train_data.head()

feats = ['AAPL',"AMZN","GOOGL"]
#,"AMZN","GOOGL"

class Env:
  def __init__(self, df):
    self.df = df
    self.n = len(df)
    self.current_idx = 0
    self.action_space = [0, 1, 2] # BUY, SELL, HOLD
    self.invested = 0

    self.states = self.df[feats].to_numpy()
    self.rewards = self.df['SPY'].to_numpy()

  def reset(self):
    self.current_idx = 0
    return self.states[self.current_idx]

  def step(self, action):
    # need to return (next_state, reward, done)

    self.current_idx += 1
    if self.current_idx >= self.n:
      raise Exception("Episode already done")

    if action == 0: # BUY
      self.invested = 1
    elif action == 1: # SELL
      self.invested = 0
    
    # compute reward
    if self.invested:
      reward = self.rewards[self.current_idx]
    else:
      reward = 0

    # state transition
    next_state = self.states[self.current_idx]

    done = (self.current_idx == self.n - 1)
    return next_state, reward, done

class StateMapper:
  def __init__(self, env, n_bins=6, n_samples=10000):
    # first, collect sample states from the environment and then we are going to create bin based on these states
    states = []
    done = False
    s = env.reset()
    self.D = len(env.states) # number of elements we need to bin, number of section in reward [0.1765] = 1
    #print(self.D)
    states.append(s)
    for _ in range(n_samples):
      a = np.random.choice(env.action_space)
      s2, _, done = env.step(a)
      states.append(s2)
      if done:
        s = env.reset()
        states.append(s)

    # convert to numpy array for easy indexing
    states = np.array(states)
    
    # create the bins for each dimension
    self.bins = []
    for d in range(self.D):
        
      #states[:,d] - means column, slicing (0)  
      column = np.sort(states[:,d])

      # find the boundaries for each bin
      current_bin = []
      for k in range(n_bins):
        boundary = column[int(n_samples / n_bins * (k + 0.5))]
        current_bin.append(boundary)

      self.bins.append(current_bin)
    #print(self.bins)

  def transform(self, state):
    x = np.zeros(self.D)
    for d in range(self.D):
      x[d] = int(np.digitize(state[d], self.bins[d])) #(6.0,)
    #print(x)
    return tuple(x)

  #Create iterative tool to specify the bins
  def all_possible_states(self):
    list_of_bins = []
    for d in range(self.D):
      list_of_bins.append(list(range(len(self.bins[d]) + 1)))
    return itertools.product(*list_of_bins)

class Agent:
  def __init__(self, action_size, state_mapper):
    self.action_size = action_size
    self.gamma = 0.8  # discount factor
    self.epsilon = 0.1
    self.learning_rate = 1e-1
    self.state_mapper = state_mapper

    # initialize Q-table randomly
    self.Q = {}
    for s in self.state_mapper.all_possible_states():
      print(s)
      s = tuple(s)
      for a in range(self.action_size):
        self.Q[(s,a)] = np.random.randn()


  def act(self, state):
    if np.random.rand() <= self.epsilon:
      return np.random.choice(self.action_size)

    s = self.state_mapper.transform(state)
    act_values = [self.Q[(s,a)] for a in range(self.action_size)]
    return np.argmax(act_values)  # returns action

  def train(self, state, action, reward, next_state, done):
    s = self.state_mapper.transform(state)
    s2 = self.state_mapper.transform(next_state)

    if done:
      target = reward
    else:
      act_values = [self.Q[(s2,a)] for a in range(self.action_size)]
      target = reward + self.gamma * np.amax(act_values)

    # Run one training step
    self.Q[(s,action)] += self.learning_rate * (target - self.Q[(s,action)])

def play_one_episode(agent, env, is_train):
  state = env.reset()
  print(state)
  done = False
  total_reward = 0
  action_list = []
  
  
  while not done:
    action = agent.act(state)
    next_state, reward, done = env.step(action)
    total_reward += reward
    action_list.append(env.invested)
    if is_train:
      agent.train(state, action, reward, next_state, done)
    state = next_state
  
  return total_reward,action_list

num_episodes = 700
action_list = []

train_env = Env(train_data)
test_env = Env(test_data)

action_size = len(train_env.action_space)
state_mapper = StateMapper(train_env)
agent = Agent(action_size, state_mapper)

train_rewards = np.empty(num_episodes)
test_rewards = np.empty(num_episodes)

for e in range(num_episodes):
  r,a = play_one_episode(agent, train_env, is_train=True)
  train_rewards[e] = r

  # test on the test set
  tmp_epsilon = agent.epsilon
  agent.epsilon = 0.
  tr,a = play_one_episode(agent, test_env, is_train=False)
  agent.epsilon = tmp_epsilon
  test_rewards[e] = tr

  print(f"eps: {e + 1}/{num_episodes}, train: {r:.5f}, test: {tr:.5f}")
  
tr,action_list = play_one_episode(agent, test_env, is_train=False)
plt.plot(train_rewards)
plt.plot(test_rewards);