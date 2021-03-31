import gym
import math
import numpy as np
import time
import random

from gym_ssl.grsim_ssl.grSimSSL_env import GrSimSSLEnv
from gym_ssl.grsim_ssl.Communication.grSimClient import grSimClient
from gym_ssl.grsim_ssl.Entities import Ball, Frame, Robot
from gym_ssl.grsim_ssl.GoalieEnv import goalieState
from gym_ssl.grsim_ssl.GoalieEnv import LEFT_GOALY, RIGHT_GOALY
from gym_ssl.grsim_ssl.Utils import mod


class goalieEnv(GrSimSSLEnv):
    """
    Description:
    # Goalie Env, the expert GK
    Source:
    # TODO

    Observation:
        Type: Box(13)
        Num     Observation                                       Min                     Max
        0       Rel Ball X   (mm)                               -7000                   7000
        1       Rel Ball Y   (mm)                               -6000                   6000
        2       Ball Vx  (mm/s)                                 -10000                  10000
        3       Ball Vy  (mm/s)                                 -10000                  10000
        4       Blue id 0 Vx  (mm/s)                            -10000                  10000
        5       Blue id 0 Vy  (mm/s)                            -10000                  10000
        6       Blue id 0 Robot Theta    (rad)                  -math.pi                math.pi 
        7       Blue id 0 Robot Vw       (rad/s)                -math.pi * 3            math.pi * 3
        8       Dist Blue id0 - ball (mm)                       -5000                   5000
        9       Angular Dist Blue id0 - ball (mm)               -math.pi                math.pi
        10      Dist X Blue id0 - goal centerX (mm)             -2000                   2000
        11      Dist Y Blue id0 - goal left (mm)                -2000                   2000 
        12      Dist Y Blue id0 - goal right (mm)               -2000                   2000 
    
    Actions:
        Type: Box(3)
        Num     Action                        Min                     Max
        0       Blue id 0 Vx                  -5                      5
        1       Blue id 0 Vy                  -5                      5
        2       Blue id 0 Vw (rad/s)        -math.pi * 3            math.pi * 3
    
    Reward:
        +1 for success (+0.5 bonus if catch)
        -1 to fails
        0 otherwise.

    Starting State:
    All observations are assigned a uniform random value in [-0.05..0.05]
    # TODO

    Episode Termination:
    # TODO
    """
    def __init__(self, rot_goalie=False):
        super().__init__()
        ## Action Space
        self.rot_goalie = rot_goalie
        actSpaceThresholds = np.array([1, 1, math.pi], dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-actSpaceThresholds, high=actSpaceThresholds)

        # Observation Space thresholds
        obsSpaceThresholds = np.array([7000, 6000, 10000, 10000,
                                       10000, 10000, math.pi, math.pi * 3, 
                                       5000, math.pi, 2000, 2000 , 2000], dtype=np.float32)
        
        self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)
        self.goalieState = None
        self.target_bally = 0
        self.start = time.time()
        self.good_reward = [1, 1.5]
        self.start_angle = 90 if self.rot_goalie else 0
        print(self.rot_goalie, self.start_angle)

        print('Environment initialized')

    def _getCommands(self, actions):
        commands = []
        cmdAttacker = Robot(id=0, yellow=False, vx=actions[0], vy=actions[1], vw=actions[2], dribbler=True)

        commands.append(cmdAttacker)

        # Moving Attacker
        ## TODO
        # Moving Ball 
        ## TODO
        #ball = self._getCorrectBallCommand()
        #commands.append(ball)
        return commands

    def _parseObservationFromState(self):
        observation = []

        self.goalieState = goalieState()
        observation = self.goalieState.getObservation(self.state)

        return np.array(observation)

    def reset(self):
        # Remove ball from Robot
        self.client.sendCommandsPacket([Robot(yellow=False, id = 0, kickVx=0, theta=self.start_angle), Robot(yellow=True, id = 0, kickVx=3)]) 
        self.client.receiveState()
        self.start = time.time()
        
        return super().reset()

    def _getFormation(self):
        attacker_x = -4
        attacker_y = random.randrange(-20, 20, 2)/10
        robot_theta = random.randrange(0, 359, 5)
        
        target_bally = np.random.uniform(-0.59, 0.59)/10
        ball_kp    = np.random.uniform(4.0, 6.0)
        
        ball_theta = np.random.uniform(-np.pi/2*0.9, np.pi/2*0.9)
        ball_d     = np.random.uniform(1.2, 5)
        ball_x =  -6 + (np.cos(ball_theta) * ball_d)
        ball_y = np.sin(ball_theta) * ball_d
        ball_y = min(ball_y, 4.5)
        ball_y = max(ball_y, -4.5)
        

        #ball_x = random.randrange(-3, 0)#0.1*math.cos(math.radians(robot_theta)) + attacker_x
        #ball_y = random.randrange(-3, 3)#0.1*math.sin(math.radians(robot_theta)) + attacker_y
        
        
        # ball position
        ball = Ball(x=ball_x, y=ball_y, vx=0, vy=0)
        ball_vx, ball_vy = self._getBallSpeeds(ball_x, ball_y, target_bally, ball_kp)
        
        ball.vx = ball_vx
        ball.vy = ball_vy
        
        # Goalkeeper position
        goalkeeper_y = random.randrange(-5, 5, 1)/10
        goalKeeper = Robot(id=0, x=-6, y=goalkeeper_y, theta=self.start_angle, yellow = False)
        # Kicker position
        attacker = Robot(id=0, x=0, y=0, theta=robot_theta, yellow=True)

        # For fixed positions!
        # ball = Ball(x=-4.1, y=0, vx=0, vy=0)
        # # Goalkeeper position
        # goalKeeper = Robot(id=0, x=-6, y=0, theta=0, yellow = True)
        # # Kicker position
        # attacker = Robot(id=0, x=-4, y=0, theta=180, yellow = False)

        return [goalKeeper, attacker], ball

    def _calculateRewardsAndDoneFlag(self):
        return self._goalieRewardFunction()
    
    def _goalieRewardFunction(self):
        reward = 0
        done = False
        
        # GK out of big area
        if self.state.robotsBlue[0].x > -4800 or abs(self.state.robotsBlue[0].y) > 1200:
            reward -= 0.01
            
        # GK control the ball
        if self.goalieState.robot_ball_dist < 120 and abs(self.goalieState.angle_relative)<0.5:    
            done = True
            reward = 1.5
            return reward, done
        
        # the ball out the field limits
        elif self.state.ball.x < -6000:
            done = True
          
            # ball entered the goal
            if self.state.ball.y < 600 and self.state.ball.y > -600:
                reward = -1
                
            # ball wrong or gk defender 
            else:
                reward =  1
        
        # If ball is moving away from goal after attacker kick NOT GOAL
        elif self.state.ball.x < -5000:
            if self.state.ball.vx > 0:
                done = True
                reward = 1
                
        # timeout and not goal
        elif time.time() - self.start > 30:
            done = True
            reward = 1
                

        return reward, done


    def _getBallSpeeds(self, ball_x, ball_y, ball_yt, ball_kp):
        
        
        errX = -6 - ball_x
        errY = ball_yt  - ball_y
        dist = ((errX**2) + (errY**2))**(1/2)
        
        
        vx = ball_kp*(errX/dist)
        vy = ball_kp*(errY/dist)
        return vx, vy
        
        
        
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