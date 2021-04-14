import numpy as np
import random
import gym
from gym import wrappers
import matplotlib.pyplot as plt
import time
import os

def get_action(s, w):
    return 1 if s.dot(w) > 0 else 0

def play_one_episode(env,new_params):
    done = False
    observation = env.reset()
    t = 0
    
    while not done or t < 10000:
        env.render()
        t += 1
        action = get_action(observation,new_params)
        observation, reward, done, _ = env.step(action)
        if done:
            break
        time.sleep(0.01)
    return t

def multiple_episodes(env,params):
    avg = np.empty(100)
    for i in range(100):
        avg[i] = play_one_episode(env,params)
    
    mean = avg.mean()
    return mean

def random_search(env):
    best = 0
    params = None
    overall_mean = []
    rounds = 0
    
    for i in range(10):
        new_params = np.random.random(4)*2 - 1 
        avg = multiple_episodes(env,new_params)
        overall_mean.append(avg)
    
        if best < avg:
            best = avg 
            params = new_params
            rounds = i

    return overall_mean,params,rounds

if __name__ == "__main__":
    env = gym.make("CartPole-v0")
    env = wrappers.Monitor(env,os.getcwd())
    avg_of_rounds, params,rounds = random_search(env)
    plt.plot(avg_of_rounds)
    plt.show()
    
    print(params,rounds)
    
    