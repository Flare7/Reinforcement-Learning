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

class Env:
    def __init__(self,df):
        self.df = df
        self.current_idx = 0
        self.state_space = [0,1,2]
        self.invested = 0
        self.n = len(df)
        
        self.rewards = self.df["SPY"].to_numpy()
        self.states = self.df[feats].to_numpy()
        
    def reset(self):
        self.current_idx = 0
        return self.states[self.current_idx]
    
    def step(self,action):
        self.current_idx += 1
        
        if self.current_idx >= self.n:
            raise "DONE"
            
        if action == 0:
            self.invested = 1
        elif action == 1:
            self.invested = 0
        
        if self.invested:
            reward = self.rewards[self.current_idx]
        else:
            reward = 0
            
        next_state = self.states[self.current_idx]
        
        done = (self.current_idx == self.n -1)
        return done,next_state,reward
    
class Brain:
    def __init__(self,env,n_bins = 6,n_samples = 10000):
        self.env = env
        self.bins = []
        s = env.reset()
        self.Dimension = len(s)
        states = []
        states.append(s)
        
        for _ in range(n_samples):
            a = np.random.choice(env.state_space)
            done,s2,r = env.step(a)
            states.append(s2)
            if done:
                s = env.reset()
                states.append(s)
        
        states = np.array(states)
        
        for d in range(self.Dimension):
            
            columns = np.sort(states[:,d])
            
            current_bin = []
            for k in range(n_bins):
                current_bin.append(int(n_samples/n_bins*(k*0.5)))
            
            self.bins.append(current_bin)
            
    def all_possible_states(self):
        list_of_bins = []
        for d in range(self.Dimension):
            list_of_bins.append(list(range(len(self.bins[d]+1))))
        print(list_of_bins)
        return itertools(*list_of_bins)
        
            
class Agent:
    def __init__(self,brain,env):
        self.Q = {}
        
        for s in brain.all_possible_states():
            s = tuple(s)
            for a in range(len(env.state_space)):
                self.Q[s,a] = np.random.randn()

train_env = Env(train_data)
test_env = Env(test_data)

brain_obj = Brain(train_env)
agent = (brain_obj,train_env)

    