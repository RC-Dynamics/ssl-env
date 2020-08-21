import gym
import gym_ssl
import numpy as np
import os
import sys

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch

from agents.Utils.Normalization import NormalizedWrapper
from agents.Utils.Networks import ValueNetwork, PolicyNetwork
from agents.Utils.OUNoise import OUNoise
from agents.Utils.ReplayBuffer import ReplayBuffer


class AgentDDPG:

    def __init__(self, name='DDPG',
                 maxEpisodes=30000, maxSteps=150, batchSize=256, replayBufferSize=200000, valueLR=1e-3, policyLR=1e-4,
                 hiddenDim=256, nEpisodesPerCheckpoint=5000):
        # Training Parameters
        self.batchSize = batchSize
        self.maxSteps = maxSteps
        self.maxEpisodes = maxEpisodes
        self.nEpisodesPerCheckpoint = nEpisodesPerCheckpoint
        self.nEpisodes = 0

        # Check if cuda gpu is available, and select it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print('Cuda Activated? ', torch.cuda.is_available())
            
        # Create Environment using a wrapper which scales actions and observations to [-1, 1]
        self.env = NormalizedWrapper(gym.make("grSimSSLShootGoalie-v0"))

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
        self.valueOptimizer = optim.Adam(self.valueNet.parameters(), lr=valueLR)
        self.policyOptimizer = optim.Adam(self.policyNet.parameters(), lr=policyLR)

        # Init replay buffer
        self.replayBuffer = ReplayBuffer(replayBufferSize)

        # Tensorboard Init
        self.path = './runs/' + name
        self.loadedModel = self._load()
        if sys.argv[2] == 'train':
            import socket
            from datetime import datetime
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(
                self.path, current_time + '_' + socket.gethostname())
            self.writer = SummaryWriter(log_dir=log_dir)
        

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
        # TODO pausar treino quando apertar algum botão e salvar estado
        while self.nEpisodes < self.maxEpisodes:
            state = self.env.reset()
            self.ouNoise.reset()
            episodeReward = 0
            nStepsInEpisode = 0

            while nStepsInEpisode < self.maxSteps:
                action = self.policyNet.get_action(state)
                action = self.ouNoise.get_action(action, nStepsInEpisode)
                next_state, reward, done, _ = self.env.step(action)

                self.replayBuffer.push(state, action, reward, next_state, done)
                if len(self.replayBuffer) > self.batchSize:
                    self._update(self.batchSize)

                state = next_state
                episodeReward = reward
                nStepsInEpisode += 1

                if done:
                    break

            self.nEpisodes += 1

            # TODO trocar por lista circular
            # rewards.append(episodeReward)

            self.writer.add_scalar('Train/Reward', episodeReward, self.nEpisodes)
            self.writer.add_scalar('Train/Steps', nStepsInEpisode, self.nEpisodes)

            # TODO arquivo separado a cada x passos
            if (self.nEpisodes % self.nEpisodesPerCheckpoint) == 0:
                torch.save({
                    'valueNetDict': self.valueNet.state_dict(),
                    'policyNetDict': self.targetPolicyNet.state_dict(),
                    'targetValueNetDict': self.targetValueNet.state_dict(),
                    'targetPolicyNetDict': self.targetPolicyNet.state_dict(),
                    'nEpisodes': self.nEpisodes
                }, self.path + '/checkpoint')

        self.writer.flush()

    # Playing loop
    def play(self):
        if self.loadedModel:
            while True:
                done = False
                obs = self.env.reset()
                while not done:
                    action = self.policyNet.get_action(obs)
                    obs, reward, done, _ = self.env.step(action)
        else:
            print("Correct usage: python train.py {name} (play | train)")

    def _load(self):
        # Check if checkpoint file exists
        if os.path.exists(self.path + '/checkpoint'):
            checkpoint = torch.load(self.path + '/checkpoint')
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

if __name__ == '__main__':

    if len(sys.argv) == 3:
        agent = AgentDDPG(name=sys.argv[1])
        if sys.argv[2] == 'play':
            agent.play()
        elif sys.argv[2] == 'train':
            agent.train()
        else:
            print("Correct usage: python train.py {name} (play | train)")
    else:
        print("Correct usage: python train.py {name} (play | train)")