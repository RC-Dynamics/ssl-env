import gym
import gym_ssl
import numpy as np
import os
import sys
import time

from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
import torch
import argparse
from agents.Utils.Normalization import NormalizedWrapper
from agents.Utils.Networks import TD3ValueNetwork, TD3PolicyNetwork
from agents.Utils.OUNoise import OUNoise
from agents.Utils.Buffers import ReplayBuffer, AverageBuffer
from tqdm import tqdm
import copy
from pathlib import Path

decay_period=500_000
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, stateDim, actionDim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(stateDim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, actionDim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).cuda()
        action = self.forward(state)
        return action.detach().cpu().numpy()[0]



class Critic(nn.Module):
    def __init__(self, stateDim, actionDim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(stateDim + actionDim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(stateDim + actionDim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)


    def forward(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2


    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1
            

class AgentTD3:

    def __init__(self, name='TD3',
                 maxEpisodes=60001, maxSteps=150, batchSize=256, replayBufferSize=1_000_000, valueLR=3e-4, policyLR=3e-4,
                 hiddenDim=256, nEpisodesPerCheckpoint=5000, ckpt_stem='checkpoint', env_str="grSimSSLShootGoalie-v01", loss_f=None):
        # Training Parameters

        self.batchSize   = batchSize
        self.maxSteps    = maxSteps
        self.maxEpisodes = maxEpisodes
        self.nEpisodesPerCheckpoint = nEpisodesPerCheckpoint
        self.nEpisodes   = 0
        self.ckpt_stem = r'/' + ckpt_stem
        self.name = name
        self.env_str = env_str

        # Check if cuda gpu is available, and select it
        self.device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
            
        # Create Environment using a wrapper which scales actions and observations to [-1, 1]
        self.env = NormalizedWrapper(gym.make(self.env_str))
        self.max_action = float(self.env.action_space.high[0])

        # Init action noise object
        #self.noise = GaussianExploration(self.env.action_space)
        #self.ouNoise = OUNoise(self.env.action_space)

        # Init networks
        stateDim = self.env.observation_space.shape[0]
        actionDim = self.env.action_space.shape[0]
        self.actionDim = actionDim

        self.actor = Actor(stateDim, actionDim, self.max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = Critic(stateDim, actionDim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

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
        self.value_criterion = loss_f()

        self.policy_noise  = 0.2
        self.expl_noise    = 0.1
        self.noise_std     = 0.2
        self.noise_clip    = 0.5
        self.policy_update = 2
        self.tau           = 0.005

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()


    def _update(self, step, batch_size,
                gamma=0.99,
                min_value=-np.inf,
                max_value=np.inf,
                soft_tau=1e-2):

        
        state, action, reward, next_state, done = self.replayBuffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Get and modify next action
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            #next_action  = self.targetPolicyNet(next_state)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action, self.max_action)


            # The lesser of two evils 
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + (1.0 - done) * gamma * target_Q

        current_Q1, current_Q2 = self.critic(state, action)

        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if step % self.policy_update == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)



    # Training Loop
    def train(self):
        self.env.set_seed(2021)
        for self.nEpisodes in tqdm(range(self.maxEpisodes)):
            state = self.env.reset()
            #self.ouNoise.reset()
            episodeReward = 0
            nStepsInEpisode = 0
            stepSeg = -1
            self.startTimeInEpisode = time.time()

            while nStepsInEpisode < self.maxSteps:
                decay_step = (nStepsInEpisode + ((self.nEpisodes+1) * self.maxSteps))

                if self.nEpisodes <= (self.maxEpisodes * 0.001):
                    #print(self.nEpisodes, 'Sample')
                    action = self.env.action_space.sample()
                elif self.nEpisodes >= (self.maxEpisodes * 0.9):
                    action = self.select_action(np.array(state))
                else:
                    action = (self.select_action(np.array(state)) + 
                              np.random.normal(0, self.max_action * self.expl_noise, size=self.actionDim)
                             ).clip(-self.max_action, self.max_action)
                
                next_state, reward, done, info = self.env.step(action)

                self.replayBuffer.push(state, action, reward, next_state, done)
                if len(self.replayBuffer) > self.batchSize:
                    self._update(nStepsInEpisode, self.batchSize)

                state            = next_state
                episodeReward   += reward
                nStepsInEpisode += 1

                if done:
                    self.goalsBuffer.push(1 if reward > 0 else 0)
                    break   
            

            if nStepsInEpisode > 1:
                stepSeg = nStepsInEpisode/(time.time() - self.startTimeInEpisode)

            self.rewardsBuffer.push(episodeReward)

            for key in info:
                self.writer.add_scalar(f'Train/{key}', info[key], self.nEpisodes)  

            self.writer.add_scalar('Train/Reward', episodeReward, self.nEpisodes)
            self.writer.add_scalar('Train/Steps', nStepsInEpisode, self.nEpisodes)
            self.writer.add_scalar('Train/Goals_average_on_{}_previous_episodes'.format(self.goalsBuffer.capacity), self.goalsBuffer.average(), self.nEpisodes)
            self.writer.add_scalar('Train/Steps_seconds',stepSeg, self.nEpisodes)
            self.writer.add_scalar('Train/Reward_average_on_{}_previous_episodes'.format(self.rewardsBuffer.capacity), self.rewardsBuffer.average(), self.nEpisodes)
            self.writer.add_scalar('Train/NoiseS', min(1.0, decay_step / decay_period), self.nEpisodes)

            if (((self.nEpisodes) % self.nEpisodesPerCheckpoint) == 0) and (self.nEpisodes != 0):
                self._save()

        self.writer.flush()

    # Playing loop
    def play(self):
        steps_hist  = []
        rewards     = [] 
        infer_times = []
        n_iter      = 1000
        base_idx    = 1_000_000
        logger_file_name = f'logger_result_{self.name}_{self.env_str}.txt'
        if self.loadedModel:
            for run in tqdm(range(n_iter)):
                done = False
                self.env.set_seed(base_idx + run)
                obs = self.env.reset()

                self.env.step(
                    np.zeros_like(self.env.action_space.shape[0]))
                
                time.sleep(0.001)
                steps = 0
                infer_time = []
                while not done and steps < self.maxSteps:
                    steps += 1
                    start_time = time.time()
                    action = self.select_action(np.array(obs)) 
                    end_time   = time.time()
                    obs, reward, done, _ = self.env.step(action)
                    infer_time.append(end_time - start_time)
                steps_hist.append(steps)
                rewards.append(reward)
                infer_time.append(np.array(infer_time).mean())
            
            infer_time = np.array(infer_time)
            steps_hist = np.array(steps_hist)

            header     = "env,run,ckpt,n,mean steps,std steps,mean time,std time,"
            header    += ','.join(list(map(lambda x:str(x), self.env.good_reward)))

            save_str = f"{self.env_str},{self.name},{self.ckpt_stem},{n_iter},{steps_hist.mean()}, {steps_hist.std()}, {infer_time.mean()}, {infer_time.std()},"
            id_unique, freq = np.unique(rewards, return_counts=True)

            for idx in self.env.good_reward:
                if idx in id_unique:
                    val = freq[np.where(id_unique==idx)[0][0]]
                else:
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
            print("Model not load")

    def _load(self):
        # Check if checkpoint file exists
        if os.path.exists(self.path + self.ckpt_stem):
            checkpoint = torch.load(self.path + self.ckpt_stem, map_location=self.device)
            # Load networks parameters checkpoint
            self.actor.load_state_dict(checkpoint['actorDict'])
            self.actor_target.load_state_dict(checkpoint['targetActorDict'])
            self.critic.load_state_dict(checkpoint['criticDict'])
            self.critic_target.load_state_dict(checkpoint['targetCriticDict'])
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
            'actorDict': self.actor.state_dict(),
            'targetActorDict': self.actor_target.state_dict(),
            'criticDict': self.critic.state_dict(),
            'targetCriticDict': self.critic_target.state_dict(),
            'nEpisodes': self.nEpisodes,
            'goalsBuffer': self.goalsBuffer.state_dict(),
            'rewardsBuffer': self.rewardsBuffer.state_dict()
        }, self.path + '/checkpoint')

        torch.save({
            'actorDict': self.actor.state_dict(),
            'targetActorDict': self.actor_target.state_dict(),
            'criticDict': self.critic.state_dict(),
            'targetCriticDict': self.critic_target.state_dict(),
            'nEpisodes': self.nEpisodes,
            'goalsBuffer': self.goalsBuffer.state_dict(),
            'rewardsBuffer': self.rewardsBuffer.state_dict()
        }, self.path + '/checkpoint_' + str(self.nEpisodes))


def arg_parser():
    """Arg parser"""    
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--name', type=str,
                        default='TD3_0000', help='exp name')
    parser.add_argument('-a', '--auto', action='store_true',
                        help='exp name auto')
    parser.add_argument('-f', '--funct', type=str,
                        default='train', help='train | play | full-play')
    parser.add_argument('-l', '--loss', type=str,
                        default='mse', help='mse | l1 | bce | bcel')
    parser.add_argument('-e', '--env', type=str,
                        default='grSimSSLShootGoalie-v01',
                        help='path to dataset')
    return parser

def get_loss(label):
    if label == 'mse':
        return nn.MSELoss
    if label == 'l1':
        return nn.SmoothL1Loss
    if label == 'bce':
        return nn.BCELoss
    if label == 'bcelogits':
        return nn.BCEWithLogitsLoss

if __name__ == '__main__':
    parser    = arg_parser()
    args      = vars(parser.parse_args())

    name      = args['name']
    funct     = args['funct']
    env       = args['env']
    loss_l    = args['loss']
    loss_f    = get_loss(loss_l)
    
    print('\nArgs:')
    [ print('\t* {}: {}'.format(k,v) ) for k,v in (args).items() ]
       

    if funct == 'play':
        agent = AgentTD3(name=name, env_str=env, loss_f=loss_f)
        agent.play()

    if funct == 'train': 
        if args['auto']:
            name = f'{env}_TD3_{loss_l}_{name}'

        path = './runs/' + name
        Path(path).mkdir(exist_ok=True, parents=True)


        with open(f"{path}/config.txt", 'a') as f:
            write_line = '\n'.join(['\t* {}: {}'.format(k,v)  for k,v in (args).items() ])
            f.write(write_line + '\n')

        agent = AgentTD3(name=name, env_str=env, loss_f=loss_f)
        agent.train()

    if funct == 'full-play' or funct == 'train':

        path = './runs/' + name
        for stem in sorted(glob(path + r'/checkpoint*')):
            stem = Path(stem).stem
            agent = AgentTD3(name=name, ckpt_stem=stem, env_str=env, loss_f=loss_f)
            agent.play()
