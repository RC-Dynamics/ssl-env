import gym
import gym_ssl
import numpy as np

from stable_baselines3 import DDPG

env = gym.make('grSimSSLGoToBall-v0')

model = DDPG.load("./models/gotoball3")

while True:
    done = False
    obs = env.reset()
    while not done:
        action, _states = model.predict(obs)
        obs, rewards, done, info = env.step(action)