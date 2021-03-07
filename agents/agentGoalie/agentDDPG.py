import gym
import gym_ssl
import numpy as np
import os
import sys
import time
import datetime
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch

from agents.Utils.Normalization import NormalizedWrapper
from agents.Utils.Networks import ValueNetwork, PolicyNetwork
from agents.Utils.OUNoise import OUNoise
from agents.Utils.Buffers import ReplayBuffer, AverageBuffer


class AgentDDPG:

    def __init__(self, name='DDPG',
                 maxEpisodes=60000, maxSteps=150, batchSize=256, replayBufferSize=1_000_000, valueLR=1e-4, policyLR=1e-4,
                 hiddenDim=256, nEpisodesPerCheckpoint=5000):
        # Training Parameters
        self.batchSize   = batchSize
        self.maxSteps    = maxSteps
        self.maxEpisodes = maxEpisodes
        self.nEpisodesPerCheckpoint = nEpisodesPerCheckpoint
        self.nEpisodes = 0

        # Check if cuda gpu is available, and select it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Cuda Activated? ', torch.cuda.is_available())
            
        # Create Environment using a wrapper which scales actions and observations to [-1, 1]
        self.env = NormalizedWrapper(gym.make("grSimSSLGK-v0"))

        # Init action noise object
        self.ouNoise = OUNoise(self.env.action_space)

        # Init networks
        stateDim = self.env.observation_space.shape[0]
        actionDim = self.env.action_space.shape[0]
        self.valueNet = ValueNetwork(stateDim, actionDim, hiddenDim).to(self.device)
        self.policyNet = PolicyNetwork(stateDim, actionDim, hiddenDim, device=self.device).to(self.device)
        self.targetValueNet = ValueNetwork(stateDim, actionDim, hiddenDim).to(self.device)
        self.targetPolicyNet = PolicyNetwork(stateDim, actionDim, hiddenDim, device=self.device).to(self.device)
        # Same initial parameters for target networks
        for target_param, param in zip(self.targetValueNet.parameters(), self.valueNet.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.targetPolicyNet.parameters(), self.policyNet.parameters()):
            target_param.data.copy_(param.data)

        # Init optimizers
        self.valueOptimizer = optim.RMSprop(self.valueNet.parameters(), lr=valueLR)
        self.policyOptimizer = optim.RMSprop(self.policyNet.parameters(), lr=policyLR)

        # Init replay buffer
        self.replayBuffer = ReplayBuffer(replayBufferSize)

         # Init goals buffer
        self.goalsBuffer = AverageBuffer()

        # Init rewars buffer
        self.rewardsBuffer = AverageBuffer()

        # Steps per Seconds Parameters
        self.startTimeInEpisode  = 0.0

        # Tensorboard Init
        self.path = './runs/' + name
        self.loadedModel = self._load()
        self.writer = SummaryWriter(log_dir=self.path)
        

    def _update(self, batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):
        state, action, reward, next_state, done = self.replayBuffer.sample(batch_size)

        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        policyLoss = self.valueNet(state, self.policyNet(state))
        policyLoss = -policyLoss.mean()

        next_action = self.targetPolicyNet(next_state)
        target_value = self.targetValueNet(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.valueNet(state, action)
        value_criterion = nn.MSELoss()
        value_loss = value_criterion(value, expected_value.detach())

        self.policyOptimizer.zero_grad()
        policyLoss.backward()
        self.policyOptimizer.step()

        self.valueOptimizer.zero_grad()
        value_loss.backward()
        self.valueOptimizer.step()

        for target_param, param in zip(self.targetValueNet.parameters(), self.valueNet.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

        for target_param, param in zip(self.targetPolicyNet.parameters(), self.policyNet.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - soft_tau) + param.data * soft_tau
            )

    # Training Loop
    def train(self):
        for self.nEpisodes in tqdm(range(self.maxEpisodes)):
            state = self.env.reset()
            self.ouNoise.reset()
            episodeReward = 0
            nStepsInEpisode = 0
            stepSeg = -1
            self.startTimeInEpisode = time.time()

            while nStepsInEpisode < self.maxSteps:
                action = self.policyNet.get_action(state)
                action = self.ouNoise.get_action(action, nStepsInEpisode)
                
                
                next_state, reward, done, _ = self.env.step(action)

                self.replayBuffer.push(state, action, reward, next_state, done)
                if len(self.replayBuffer) > self.batchSize:
                    self._update(self.batchSize)

                state = next_state
                episodeReward += reward
                nStepsInEpisode += 1

                if done:
                    self.goalsBuffer.push(1 if reward > 0 else 0)
                    break   
            

            if nStepsInEpisode > 1:
                stepSeg = nStepsInEpisode/(time.time() - self.startTimeInEpisode)

            self.rewardsBuffer.push(episodeReward)
            
            # TODO trocar por lista circular
            # rewards.append(episodeReward)

            self.writer.add_scalar('Train/Reward', episodeReward, self.nEpisodes)
            self.writer.add_scalar('Train/Steps', nStepsInEpisode, self.nEpisodes)
            self.writer.add_scalar('Train/Goals_average_on_{}_previous_episodes'.format(self.goalsBuffer.capacity), self.goalsBuffer.average(), self.nEpisodes)
            self.writer.add_scalar('Train/Steps_seconds',stepSeg, self.nEpisodes)
            self.writer.add_scalar('Train/Reward_average_on_{}_previous_episodes'.format(self.rewardsBuffer.capacity), self.rewardsBuffer.average(), self.nEpisodes)

            if (self.nEpisodes % self.nEpisodesPerCheckpoint) == 0:
                self._save()



        self.writer.flush()

    # Playing loop
    def play(self):
        if self.loadedModel:
            while True:
                done = False
                obs = self.env.reset()
                steps = 0
                while not done and steps < self.maxSteps:
                    steps += 1
                    action = self.policyNet.get_action(obs)
                    obs, reward, done, _ = self.env.step(action)
                time.sleep(0.1)
        else:
            print("Correct usage: python train.py {name} (play | train) [-cs]")

    def _load(self):
        # Check if checkpoint file exists
        if os.path.exists(self.path + '/checkpoint'):
            checkpoint = torch.load(self.path + '/checkpoint', map_location=self.device)
            # Load networks parameters checkpoint
            self.valueNet.load_state_dict(checkpoint['valueNetDict'])
            self.policyNet.load_state_dict(checkpoint['policyNetDict'])
            self.targetValueNet.load_state_dict(checkpoint['targetValueNetDict'])
            self.targetPolicyNet.load_state_dict(checkpoint['targetPolicyNetDict'])
            # Load number of episodes on checkpoint
            self.nEpisodes = checkpoint['nEpisodes']
            print("Checkpoint with {} episodes successfully loaded".format(self.nEpisodes))
            return True
        else:
            print("- No checkpoint " + self.path + '/checkpoint' + " loaded!")
            return False

    def _save(self):
        print("Save network parameters in episode ", self.nEpisodes)
        torch.save({
            'valueNetDict': self.valueNet.state_dict(),
            'policyNetDict': self.targetPolicyNet.state_dict(),
            'targetValueNetDict': self.targetValueNet.state_dict(),
            'targetPolicyNetDict': self.targetPolicyNet.state_dict(),
            'nEpisodes': self.nEpisodes,
            'goalsBuffer': self.goalsBuffer.state_dict(),
            'rewardsBuffer': self.rewardsBuffer.state_dict()
        }, self.path + '/checkpoint')

        torch.save({
            'valueNetDict': self.valueNet.state_dict(),
            'policyNetDict': self.targetPolicyNet.state_dict(),
            'targetValueNetDict': self.targetValueNet.state_dict(),
            'targetPolicyNetDict': self.targetPolicyNet.state_dict(),
            'nEpisodes': self.nEpisodes,
            'goalsBuffer': self.goalsBuffer.state_dict(),
            'rewardsBuffer': self.rewardsBuffer.state_dict()
        }, self.path + '/checkpoint_' + str(self.nEpisodes))


if __name__ == '__main__':
    try:
        if len(sys.argv) >= 3:
            agent = AgentDDPG(name=sys.argv[1])
            if sys.argv[2] == 'play':
                agent.play()
            elif sys.argv[2] == 'train':
                agent.train()
            else:
                print("correct usage: python train.py {name} (play | train) [-cs]")
        else:
            print("correct usage: python train.py {name} (play | train) [-cs]")
    except KeyboardInterrupt:
        if len(sys.argv) >= 4 and sys.argv[3] == '-cs':
            agent._save()