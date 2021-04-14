import gym
import numpy as np 
import random
import os
import matplotlib.pyplot as plt
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from mpl_toolkits.mplot3d import Axes3D
from gym import wrappers


class SGDRegressor:
    def __init__(self,D):
        self.w = np.random.randn(D) / np.sqrt(D)
        self.lr = 0.1
        
    def partial_fit(self, X, Y):
        #Gradient descent 
        self.w += self.lr*(Y - X.dot(self.w)).dot(X)

    def predict(self, X):
        return X.dot(self.w)
    
class FeatureTransoformer:
    def __init__(self,env):
        actions_to_take = [env.action_space.sample() for x in range(10000)]
        feature_observation = []
                
        s = env.reset()
        for i in range(1,len(actions_to_take)):
            feature_observation.append(s)
            s, reward, done, info = env.step(actions_to_take[i])
            if done:
                s = env.reset()
                done = False
        
        scaler = StandardScaler()
        scaler.fit(feature_observation)
        
        feature = FeatureUnion([
            ("rbf1",RBFSampler(gamma = 0.05, n_components = 1000)),
            ("rbf2",RBFSampler(gamma = 1.0, n_components = 1000)),
            ("rbf3",RBFSampler(gamma = 0.5, n_components = 1000)),
            ("rbf4",RBFSampler(gamma = 0.1, n_components = 1000))
            ])
        
        example_features = feature.fit_transform(scaler.transform(feature_observation))
      
        self.dimensions = example_features.shape[1]
        self.scaler = scaler
        self.featurer = feature
        
    def transform(self,observations):
        return self.featurer.transform(self.scaler.transform(observations))
    
class Model:
    def __init__(self,ft,env):
        self.ft = ft
        self.env = env
        self.models = []
        for i in range(env.action_space.n):
            model = SGDRegressor(ft.dimensions)
            self.models.append(model)
            
    def predict(self,s):
        s = self.ft.transform(np.atleast_2d(s))
        all_predictions = np.stack([m.predict(s) for m in self.models]).T
        return all_predictions
            
    def sample_action(self,s,eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:
            #argmax -> Q-learning 
            return np.argmax(self.predict(s))
        
    def update(self,s,a,G):
        s = self.ft.transform(np.atleast_2d(s))
        self.models[a].partial_fit(s,[G])
    
def play_one_episode(ft,env,model,eps):
    gamma = 0.99
    s = env.reset()
    done = False
    iters = 0
    totalreward = 0
    
    while not done and iters < 2000:
        action = model.sample_action(s,eps)
        prev_s = s
        s, reward, done, info = env.step(action)
        
        if done:
            reward = -200            
        
        next_pred = model.predict(s)
        
        G = reward + gamma*np.max(next_pred)
        model.update(prev_s, action, G)
        
        if reward == 1: # if we changed the reward to -200
            totalreward += reward
        iters += 1
    
    return totalreward

def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()

def main():
    env = gym.make("CartPole-v0")
    ft = FeatureTransoformer(env)
    model = Model(ft,env)
    
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename 
    env = wrappers.Monitor(env, monitor_dir,force=True)
    
    N = 600
    totalrewards = np.empty(N)
    for n in range(N):
        #decaying epsilon
        eps = 1.0/np.sqrt(n+1)
        totalreward = play_one_episode(ft, env, model, eps)
        totalrewards[n] = totalreward
        
        if (n) % 100 == 0:
            print("episode:", n, "total reward:", totalrewards[n])
    print("total steps:", totalrewards.sum())
    
    plt.plot(totalrewards)
    plt.title("REWARDS")
    plt.show()
    
    plot_running_avg(totalrewards)

if __name__ == "__main__":
    main()