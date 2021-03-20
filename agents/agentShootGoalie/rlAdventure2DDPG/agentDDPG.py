import gym
import gym_ssl
import numpy as np
import os
import sys
import time
import datetime
from tqdm import tqdm
from glob import glob
import argparse

from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch

from agents.Utils.Normalization import NormalizedWrapper
from agents.Utils.Networks import DDPGValueNetwork, DDPGPolicyNetwork
from agents.Utils.OUNoise import OUNoise
from agents.Utils.Buffers import ReplayBuffer, AverageBuffer


class AgentDDPG:

    def __init__(self, name='DDPG',
                 maxEpisodes=60001, maxSteps=150, batchSize=256, replayBufferSize=1_000_000, valueLR=1e-3, policyLR=1e-4,
                 hiddenDim=256, nEpisodesPerCheckpoint=5000, ckpt_stem='checkpoint', env_str="grSimSSLShootGoalie-v01"):
        # Training Parameters
        self.batchSize   = batchSize
        self.maxSteps    = maxSteps
        self.maxEpisodes = maxEpisodes
        self.nEpisodesPerCheckpoint = nEpisodesPerCheckpoint
        self.nEpisodes = 0
        self.ckpt_stem = r'/' + ckpt_stem
        self.name = name
        self.env_str = env_str

        # Check if cuda gpu is available, and select it
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print('Cuda Activated? ', torch.cuda.is_available())
            
        # Create Environment using a wrapper which scales actions and observations to [-1, 1]

        #  sparce_reward: True, move_goalie: True}
        self.env = NormalizedWrapper(gym.make(env_str))

        # Init action noise object
        self.ouNoise = OUNoise(self.env.action_space)

        # Init networks
        stateDim = self.env.observation_space.shape[0]
        actionDim = self.env.action_space.shape[0]
        self.valueNet = DDPGValueNetwork(stateDim, actionDim, hiddenDim).to(self.device)
        self.policyNet = DDPGPolicyNetwork(stateDim, actionDim, hiddenDim, device=self.device).to(self.device)
        self.targetValueNet = DDPGValueNetwork(stateDim, actionDim, hiddenDim).to(self.device)
        self.targetPolicyNet = DDPGPolicyNetwork(stateDim, actionDim, hiddenDim, device=self.device).to(self.device)
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

            if (((self.nEpisodes) % self.nEpisodesPerCheckpoint) == 0) and (self.nEpisodes != 0):
                self._save()

        self.writer.flush()

    # Playing loop
    def play(self):
        steps_hist  = []
        rewards     = [] 
        infer_times = []
        n_iter      = 1000
        logger_file_name = 'logger_result2.txt'
        if self.loadedModel:
            for run in tqdm(range(n_iter)):
                done = False
                obs = self.env.reset()
                time.sleep(0.005)
                steps = 0
                infer_time = []
                while not done and steps < self.maxSteps:
                    steps += 1
                    start_time = time.time()
                    action = self.policyNet.get_action(obs)
                    end_time   = time.time()
                    obs, reward, done, _ = self.env.step(action)
                    infer_time.append(end_time - start_time)
                steps_hist.append(steps)
                rewards.append(reward)
                infer_time.append(np.array(infer_time).mean())
            
            infer_time = np.array(infer_time)
            steps_hist = np.array(steps_hist)
            header   = "env,run,ckpt, n, mean steps, std steps, mean time, std time, N goals\n"
            save_str = f"{self.env_str},{self.name},{self.ckpt_stem},{n_iter},{steps_hist.mean()}, {steps_hist.std()}, {infer_time.mean()}, {infer_time.std()},"
            id_unique, freq = np.unique(rewards, return_counts=True)
            if 2 in id_unique:
                val = freq[np.where(id_unique==2)[0][0]]
            else:
                print("No goal?")
                val = 0
                
            save_str += f"{val},"

            write_header = not Path(logger_file_name).exists()

            with open(logger_file_name, 'a') as f:
                if write_header:
                    f.write(header)
                f.write(save_str+'\n')

            print(write_header, '\n', header)
            print(save_str)
                
        else:
            print("Correct usage: python train.py {name} (play | train) [-cs]")

    def _load(self):
        # Check if checkpoint file exists
        
        if os.path.exists(self.path + self.ckpt_stem):
            checkpoint = torch.load(self.path + self.ckpt_stem, map_location=self.device)
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
            print("- No checkpoint " + self.path + self.ckpt_stem + " loaded!")
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

def arg_parser():
    """Arg parser"""    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str,
                        default='DDPG_0000', help='exp name')
    parser.add_argument('-f', '--funct', type=str,
                        default='train', help='train | play | full-play')
    parser.add_argument('-e', '--env', type=str,
                        default='grSimSSLShootGoalie-v01',
                        help='path to dataset')

    return parser

if __name__ == '__main__':
    parser    = arg_parser()
    args      = vars(parser.parse_args())

    name      = args['name']
    funct     = args['funct']
    env       = args['env']
    
    print('\nArgs:')
    [ print('\t* {}: {}'.format(k,v) ) for k,v in (args).items() ]

    if funct == 'play':
        agent = AgentDDPG(name=name, env_str=env)
        agent.play()

    if funct == 'train': 


        agent = AgentDDPG(name=name, env_str=env)
        agent.train()

    if funct == 'full-play' or funct == 'train':

        path = './runs/' + name
        
        for stem in sorted(glob(path + r'/checkpoint*')):
            stem = Path(stem).stem
            agent = AgentDDPG(name=name, ckpt_stem=stem, env_str=env)
            agent.play()
