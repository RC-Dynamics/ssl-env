import gym
import gym_ssl

env = gym.make('grSimSSLPenalty-v0')

env.reset()
for i in range(1):
    done = False
    stateDict = {}
    while not done:
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        stateDict['ballX'] = next_state[0]
        stateDict['ballY'] = next_state[1]
        stateDict['ballVx'] = next_state[2]
        stateDict['ballVy'] = next_state[3]
        stateDict['blueY'] = next_state[4]
        stateDict['blueVy'] = next_state[5]
        stateDict['yellowX'] = next_state[6]
        stateDict['yellowY'] = next_state[7]
        stateDict['yellowTheta'] = next_state[8]
        stateDict['yellowVx'] = next_state[9]
        stateDict['yellowVy'] = next_state[10]
        stateDict['yellowOmega'] = next_state[11]
        print(stateDict)

