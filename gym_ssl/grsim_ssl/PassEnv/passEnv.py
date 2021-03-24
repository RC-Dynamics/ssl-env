import gym
import math
import numpy as np
import time
import random

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Entities import Ball, Frame, Robot
from gym_ssl.grsim_ssl.PassEnv import passState
from gym_ssl.grsim_ssl.Utils import mod


class passEnv(GrSimSSLEnv):

  """
  Description:
    # TODO
  Source:
    # TODO

  Observation:
    Type: Box(16)
    Num     Observation                                       Min                     Max
    0       R Ball X   (mm)                                   -7000                   7000
    1       R Ball Y   (mm)                                   -6000                   6000
    2       R Ball Vx  (mm/s)                                 -10000                  10000
    3       R Ball Vy  (mm/s)                                 -10000                  10000
    4       R Ally X   (mm)                                   -7000                   7000
    5       R Ally Y   (mm)                                   -6000                   6000
    6       R Ally Sin (rad)                                  -math.pi                math.pi
    7       R Ally Cos (rad)                                  -math.pi                math.pi
    8       R Dist Ally ball (mm)                             -7000                   7000
    9       Blue id 0 Robot Vw       (rad/s)                  -math.pi * 3            math.pi * 3

    
  Actions:
    Type: Box(2)
    Num     Action                        Min                     Max
    0       Blue id 0 Vw (rad/s)        -math.pi * 3            math.pi * 3
    1       Blue Kick Strength (m/s)        -6.5                   6.5
  Reward:
    Reward is 1 for success, -1 to fails. 0 otherwise.

  Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
    # TODO

  Episode Termination:
    # TODO
  """
  def __init__(self):
    super().__init__()
    ## Action Space
    
    actSpaceThresholds = np.array([math.pi * 3, 6.5], dtype=np.float32)
    self.action_space  = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)

    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000,
                                   7000, 6000, math.pi, 7000,math.pi,
                                   math.pi * 3], dtype=np.float32)

    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.passState = None
    self.min_dist = 500_000
    self.goalieState = 0
    self.good_reward = [1, 2, 5, 10]

    print('Environment initialized')
  
  def _getCommands(self, actions):
    commands = []
    cmdAttacker = Robot(id=0, yellow=False, vw=actions[0], kickVx=actions[1] if actions[1] > 0.2 else 0, dribbler=True)
    
    commands.append(cmdAttacker)

    return commands

  def _parseObservationFromState(self):
    observation = []

    self.passState = passState()
    observation = self.passState.getObservation(self.state)

    return np.array(observation)

  def reset(self):
    # Remove ball from Robot
    self.client.sendCommandsPacket([Robot(yellow=False, id=0, vw=0, kickVx=0, dribbler=True), 
                                    Robot(yellow=True, id=0, vw=0, kickVx=0, dribbler=True)]) 
    self.client.receiveState()
    self.min_dist = 500_000
    return super().reset()

  def _getFormation(self):
    
    attacker_x = 0
    attacker_y = 0
    robot_theta = random.randrange(0, 359, 5)
    attacker = Robot(id=0, x=attacker_x, y=attacker_y, theta=robot_theta, yellow = False)
    
    ball_x = 0.1*math.cos(math.radians(robot_theta)) + attacker_x
    ball_y = 0.1*math.sin(math.radians(robot_theta)) + attacker_y
    ball = Ball(x=ball_x, y=ball_y, vx=0, vy=0)
    
    dist = 0
    while(dist < 1 or dist > 6):
      ally_x = random.randrange(-40, 40, 2)/10
      ally_y = random.randrange(-40, 40, 2)/10
      dist   = np.abs(ally_x-attacker_x) + np.abs(ally_y-attacker_y)
      ally_theta = 180 + math.degrees(math.atan2(ally_y, ally_x))
      
    ally = Robot(id=0, x=ally_x, y=ally_y, theta=ally_theta, yellow = True)
    
    return [ally, attacker], ball
    
  def _calculateRewardsAndDoneFlag(self):
    return self._firstRewardFunction()

  def _firstRewardFunction(self):
    reward = 0
    done   = False
    dist_ally_ball    = self.passState.dist_ab
    self.min_dist     = min(self.min_dist, dist_ally_ball)
    dist_robot_ally   = self.passState.dist_ra
    dist_robot_ball   = self.passState.dist_rb
    dist_robot_ball_i = 120

    has_kick          = dist_robot_ball > dist_robot_ball_i
    timout            = self.passState.timestamp >= 4500 # 30 * 150
    ball_more_far     = dist_ally_ball > (dist_robot_ally + 10)
    wrong_ball        = self.min_dist < dist_ally_ball
    
    ## Ball is far from ally (need to turn)
    if dist_ally_ball > dist_robot_ally:
      reward -= 0.01

    
    ## Ball is really far from ally
    if dist_ally_ball > 8000:
      reward = -5
      done = True

    ## Ball reach the ally
    elif dist_ally_ball < 140: # magic number
      reward = 10
      done   = True
    
    elif (timout) or (has_kick and ball_more_far) or (has_kick and wrong_ball): 
      if timout and not has_kick:
        reward = -5
      if self.min_dist < 300:
        reward = 5
      elif self.min_dist < 500:
        reward = 2
      elif self.min_dist < 1000:
        reward = 1
      elif not ball_more_far:
        reward = -3
      else:
        reward = -5
      done   = True

    if done:
      self.info['min_dist'] = self.min_dist
    
    return reward, done

