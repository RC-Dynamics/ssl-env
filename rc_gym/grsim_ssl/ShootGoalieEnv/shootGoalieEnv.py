import gym
import math
import numpy as np
import time
import random

from rc_gym.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from rc_gym.grsim_ssl.Communication.grSimClient import grSimClient
from rc_gym.Entities import Ball, Frame, Robot
from rc_gym.grsim_ssl.ShootGoalieEnv import shootGoalieState
from rc_gym.Utils import mod


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
  def __init__(self):
    super().__init__()
    ## Action Space
    actSpaceThresholds = np.array([math.pi * 3, 6.5], dtype=np.float32)
    self.action_space = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)

    # Observation Space thresholds
    obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, math.pi * 3, 10000, math.pi, math.pi,
                                   math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi, math.pi], dtype=np.float32)
    self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
    self.shootGoalieState = None
    self.goalieState = 0

    self.goalieState = 0

    print('Environment initialized')
  
  def _getCommands(self, actions):
    commands = []
    cmdAttacker = Robot(id=0, yellow=False, v_theta=actions[0], kick_v_x=actions[1], dribbler=True)
    
    commands.append(cmdAttacker)


    # Moving GOALIE
    vy = (self.state.ball.y - self.state.robots_yellow[0].y)/1000
    if abs(vy) > 0.4:
      vy = 0.4*(vy/abs(vy))
    if self.state.robots_yellow[0].y > 500 and vy > 0:
      vy = 0
    if self.state.robots_yellow[0].y < -500 and vy < 0:
      vy = 0
      
    cmdGoalie = self._getCorrectGKCommand(vy)
    
    commands.append(cmdGoalie)

    return commands

  def reset(self):
    # Remove ball from Robot
    self.client.sendCommandsPacket([Robot(yellow=False, id = 0, kick_v_x=3), Robot(yellow=True, id = 0, kick_v_x=3)]) 
    self.client.receiveState()
    return super().reset()

  def _parseObservationFromState(self):
    observation = []

    self.shootGoalieState = shootGoalieState()
    observation = self.shootGoalieState.getObservation(self.state)

    return np.array(observation)

  def reset(self):
    # Remove ball from Robot
    self.client.sendCommandsPacket([Robot(yellow=False, id = 0, kick_v_x=0), Robot(yellow=True, id = 0, kick_v_x=3)]) 
    self.client.receiveState()
    return super().reset()

  def _getFormation(self):
    attacker_x = -4
    attacker_y = random.randrange(-20, 20, 2)/10
    robot_theta = random.randrange(0, 359, 5)
    ball_x = 0.1*math.cos(math.radians(robot_theta)) + attacker_x
    ball_y = 0.1*math.sin(math.radians(robot_theta)) + attacker_y
    # ball position
    ball = Ball(x=ball_x, y=ball_y, v_x=0, v_y=0)
    # Goalkeeper position
    goalkeeper_y = random.randrange(-5, 5, 1)/10
    goalKeeper = Robot(id=0, x=-6, y=goalkeeper_y, theta=0, yellow = True)
    # Kicker position
    attacker = Robot(id=0, x=attacker_x, y=attacker_y, theta=robot_theta, yellow = False)

    # For fixed positions!
    # ball = Ball(x=-4.1, y=0, v_x=0, vy=0)
    # # Goalkeeper position
    # goalKeeper = Robot(id=0, x=-6, y=0, theta=0, yellow = True)
    # # Kicker position
    # attacker = Robot(id=0, x=-4, y=0, theta=180, yellow = False)

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
    elif self.state.ball.x < -5000 and self.state.ball.v_x > -1:
        # goalkeeper caught the ball
      done = True
      reward = -1
    elif mod(self.state.ball.v_x, self.state.ball.v_y) < 10 and self.steps > 15: # 1 cm/s
      done = True
      reward = -1
    return reward, done

  def _penalizeRewardFunction(self):
    reward = -0.01
    done = False
    if self.state.ball.x < -6000:
      # the ball out the field limits
      done = True
      if self.state.ball.y < 600 and self.state.ball.y > -600:
          # ball entered the goal
          reward = 2
    elif self.state.ball.x < -5000 and self.state.ball.v_x > -1:
      done = True
      reward = -0.3
       
    return reward, done


  def _getCorrectGKCommand(self,v_y):
    '''Control goalkeeper v_theta and vx to keep him at goal line'''
    cmdGoalKeeper = Robot(yellow=True, id=0, v_y=v_y)

    # Proportional Parameters for Vx and Vw
    KpVx = 0.0006
    KpVw = 1
    # Error between goal line and goalkeeper
    errX = -6000 - self.state.robots_yellow[0].x
    # If the error is greater than 20mm, correct the goalkeeper
    if abs(errX) > 20:
        cmdGoalKeeper.v_x = KpVx * errX
    else:
        cmdGoalKeeper.v_x = 0.0
    # Error between the desired angle and goalkeeper angle
    errW = 0.0 - self.state.robots_yellow[0].theta
    # If the error is greater than 0.1 rad (5,73 deg), correct the goalkeeper
    if abs(errW) > 0.1:
        cmdGoalKeeper.v_theta = KpVw * errW
    else:
        cmdGoalKeeper.v_theta = 0.0

    return cmdGoalKeeper