import gym
import gym_ssl
import numpy as np
import time
# Using penalty env
import random
import numpy as np

from gym.envs.registration import register
from agents.Utils.Normalization import NormalizedWrapper

env = NormalizedWrapper(gym.make("grSimSSLShootGoalie-v0"))
env.reset()

for i in range(1000):
	next_state, reward, done, _ = env.step(np.array([0,0]))
	time.sleep(60*15)
    