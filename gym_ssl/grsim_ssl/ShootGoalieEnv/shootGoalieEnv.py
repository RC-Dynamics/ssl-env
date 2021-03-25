import gym
import math
import numpy as np
import time
import random

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Entities import Ball, Frame, Robot
from gym_ssl.grsim_ssl.ShootGoalieEnv import shootGoalieState
from gym_ssl.grsim_ssl.Utils import mod


class shootGoalieEnv(GrSimSSLEnv):

  """
  Description:
    # TODO
  Source:
    # TODO

  Observation:
    Type: Box(16)
    Num     Observation                                       Min                     Max
    0       Ball X   (mm)                                   -7000                   7000
    1       Ball Y   (mm)                                   -6000                   6000
    2       Ball Vx  (mm/s)                                 -10000                  10000
    3       Ball Vy  (mm/s)                                 -10000                  10000
    4       Blue id 0 Robot Vw       (rad/s)                -math.pi * 3            math.pi * 3
    5       Dist Blue id0 - goal center (mm)                -10000                  10000
    6       Angle between blue id 0 and goal left (rad)     -math.pi                math.pi
    7       Angle between blue id 0 and goal left (rad)     -math.pi                math.pi
    8       Angle between blue id 0 and goal right (rad)    -math.pi                math.pi
    9       Angle between blue id 0 and goal right (rad)    -math.pi                math.pi
    10      Angle between blue id 0 and goalie center(rad)  -math.pi                math.pi
    11      Angle between blue id 0 and goalie center(rad)  -math.pi                math.pi
    12      Angle between blue id 0 and goalie left (rad)   -math.pi                math.pi
    13      Angle between blue id 0 and goalie left (rad)   -math.pi                math.pi
    14      Angle between blue id 0 and goalie right (rad)  -math.pi                math.pi
    15      Angle between blue id 0 and goalie right (rad)  -math.pi                math.pi

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
  def __init__(self, sparce_reward=-1, move_goalie=-1):
    super().__init__()
    ## Action Space
    if sparce_reward == -1 or move_goalie == -1:
      exit(-1)
    
    self.sparce_reward = sparce_reward 
    self.move_goalie   = move_goalie
    actSpaceThresholds = np.array([math.pi * 3, 6.5], dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)

    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, math.pi * 3, 10000, math.pi, math.pi,
                                   math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.shootGoalieState  = None
    self.goalieState       = 0
    self.good_reward       = [2]

    print('Environment initialized')
  
  def _getCommands(self, actions):
    commands = []
    cmdAttacker = Robot(id=0, yellow=False, vw=actions[0], kickVx=actions[1], dribbler=True)
    
    commands.append(cmdAttacker)


    # Moving GOALIE
    if self.move_goalie:
      vy = (self.state.ball.y - self.state.robotsYellow[0].y)/1000
      if abs(vy) > 0.4:
        vy = 0.4*(vy/abs(vy))
      if self.state.robotsYellow[0].y > 500 and vy > 0:
        vy = 0
      if self.state.robotsYellow[0].y < -500 and vy < 0:
        vy = 0
        
      cmdGoalie = self._getCorrectGKCommand(vy)
      commands.append(cmdGoalie)

    return commands

  def _parseObservationFromState(self):
    observation = []

    self.shootGoalieState = shootGoalieState()
    observation = self.shootGoalieState.getObservation(self.state)

    return np.array(observation)

  def reset(self):
    # Remove ball from Robot
    self.client.sendCommandsPacket([Robot(yellow=False, id=0, vw=0, kickVx=0, dribbler=True), 
                                    Robot(yellow=True, id=0, vw=0, kickVx=3)]) 
    self.client.receiveState()
    return super().reset()

  def _getFormation(self):
    attacker_x  = -4
    attacker_y  = random.randrange(-20, 20, 2)/10
    robot_theta = random.randrange(0, 359, 5)
    ball_x = 0.1*math.cos(math.radians(robot_theta)) + attacker_x
    ball_y = 0.1*math.sin(math.radians(robot_theta)) + attacker_y
    # ball position
    ball = Ball(x=ball_x, y=ball_y, vx=0, vy=0)
    # Goalkeeper position
    if self.move_goalie:
      goalkeeper_y = random.randrange(-5, 5, 1)/10
    else:
      goalkeeper_y = 0
    goalKeeper = Robot(id=0, x=-6, y=goalkeeper_y, theta=0, yellow = True)
    # Kicker position
    attacker = Robot(id=0, x=attacker_x, y=attacker_y, theta=robot_theta, yellow = False)

    return [goalKeeper, attacker], ball
    
  def _calculateRewardsAndDoneFlag(self):
    return self._penalizeRewardFunction()

  def _firstRewardFunction(self):
    reward = 0
    done = False
    if self.state.ball.x < -6000:
      # the ball out the field limits
      done = True
      if self.state.ball.y < 600 and self.state.ball.y > -600:
          # ball entered the goal
          reward = 1
      else:
          # the ball went out the bottom line
          reward = -1
    elif self.state.ball.x < -5000 and self.state.ball.vx > -1:
        # goalkeeper caught the ball
      done = True
      reward = -1
    elif mod(self.state.ball.vx, self.state.ball.vy) < 10 and self.steps > 15: # 1 cm/s
      done = True
      reward = -1
    return reward, done

  def _penalizeRewardFunction(self):
    if self.sparce_reward:
      reward = 0
    else:
      reward = -0.01
    done = False
    if self.state.ball.x < -6000:
      # the ball out the field limits
      done = True
      if self.state.ball.y < 600 and self.state.ball.y > -600:
          # ball entered the goal
          reward = 2
    elif self.state.ball.x < -5000 and self.state.ball.vx > -1:
      done = True
      reward = -0.3
       
    return reward, done


  def _getCorrectGKCommand(self,vy):
    '''Control goalkeeper vw and vx to keep him at goal line'''
    cmdGoalKeeper = Robot(yellow=True, id=0, vy=vy)

    # Proportional Parameters for Vx and Vw
    KpVx = 0.0006
    KpVw = 1
    # Error between goal line and goalkeeper
    errX = -6000 - self.state.robotsYellow[0].x
    # If the error is greater than 20mm, correct the goalkeeper
    if abs(errX) > 20:
        cmdGoalKeeper.vx = KpVx * errX
    else:
        cmdGoalKeeper.vx = 0.0
    # Error between the desired angle and goalkeeper angle
    errW = 0.0 - self.state.robotsYellow[0].theta
    # If the error is greater than 0.1 rad (5,73 deg), correct the goalkeeper
    if abs(errW) > 0.1:
        cmdGoalKeeper.vw = KpVw * errW
    else:
        cmdGoalKeeper.vw = 0.0

    return cmdGoalKeeper