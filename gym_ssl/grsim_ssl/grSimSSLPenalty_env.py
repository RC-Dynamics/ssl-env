import gym
import math
import numpy as np
import gym_ssl.grsim_ssl.pb.grSim_Packet_pb2 as packet_pb2

from gym_ssl.grsim_ssl.grSimClient import grSimClient


class GrSimSSLPenaltyEnv(gym.Env):
    """
    Using cartpole env description as base example for our documentation
    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    Observation:
        Type: Box(3)
        Num     Observation                         Min                     Max
        0       Ball X   (mm)                       -7000                   7000
        1       Ball Y   (mm)                       -6000                   6000
        2       Ball Vx  (mm/s)                     -10000                  10000
        3       Ball Vy  (mm/s)                     -10000                  10000
        4       id 0 Blue Robot Y       (mm)        -6000                   6000
        5       id 0 Blue Robot Vy      (mm/s)      -10000                  10000
        6       id 0 Yellow Robot X     (mm)        -7000                   7000
        7       id 0 Yellow Robot Y     (mm)        -6000                   6000
        8       id 0 Yellow Robot Angle (rad)       -math.pi                math.pi
        9       id 0 Yellow Robot Vx    (mm/s)      -10000                  10000
        10      id 0 Yellow Robot Vy    (mm/s)      -10000                  10000
        11      id 0 Yellow Robot Vy    (rad/s)     -math.pi * 3            math.pi * 3

    Actions:
        Type: Box(1)
        Num     Action                        Min                     Max
        0       id 0 Blue Team Robot Vy       -1                      1

        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination: 
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.
    """

    def __init__(self):
        self.client = grSimClient()

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        # Observation Space thresholds
        obsSpaceThresholds = np.array([7000, 6000, 10000, 10000, 6000, 10000, 7000,
                                       math.pi, 6000, 1000, math.pi * 3], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)

        print('Environment initialized')

    def step(self, actions):
        # Generate command Packet from actions
        packet = self._generateCommandPacket(actions)
        # Send command Packet
        self.client.send(packet)

        visionData = self.client.receive()
        state = self._parseVision(visionData)

        reward, done = self._calculateRewards(visionData)

        return state, reward, done, {}

    def reset(self):
        # Generate replacement packet
        packet = packet_pb2.grSim_Packet()
        replacement = packet.replacement
        # Ball penalty position
        ball = replacement.ball
        ball.x = -4.8
        ball.y = 0
        ball.vx = 0
        ball.vy = 0

        # Goalkeeper penalty position
        goalKeeper = replacement.robots.add()
        goalKeeper.x = -6
        goalKeeper.y = 0
        goalKeeper.dir = 0
        goalKeeper.id = 0
        goalKeeper.yellowteam = False

        # Kicker penalty position
        attacker = replacement.robots.add()
        attacker.x = -4 
        attacker.y = 0
        attacker.dir = 180
        attacker.id = 0
        attacker.yellowteam = True

        self.client.send(packet)
        print('Environment reset')

    def _generateCommandPacket(self, actions):
        # actions = [vx, vy, omega] for blue robot 0
        packet = packet_pb2.grSim_Packet()
        grSimCommands = packet.commands
        grSimRobotCommand = grSimCommands.robot_commands
        grSimCommands.timestamp = 0.0
        grSimCommands.isteamyellow = False
        robot = grSimRobotCommand.add()
        robot.id = 0
        robot.kickspeedx = 0
        robot.kickspeedz = 0
        robot.veltangent = 0
        robot.velnormal = actions[0]
        robot.velangular = 0
        robot.spinner = False
        robot.wheelsspeed = False

        return packet

    def _parseVision(self, data):
        space = np.zeros(12)
        space[0] = data.detection.balls[0].x
        space[1] = data.detection.balls[0].y
        space[2] = data.detection.balls[0].vx
        space[3] = data.detection.balls[0].vy
        space[4] = data.detection.robots_blue[0].y
        space[5] = data.detection.robots_blue[0].vy
        space[6] = data.detection.robots_yellow[0].x
        space[7] = data.detection.robots_yellow[0].y
        space[8] = data.detection.robots_yellow[0].orientation
        space[9] = data.detection.robots_yellow[0].vx
        space[10] = data.detection.robots_yellow[0].vy
        space[11] = data.detection.robots_yellow[0].vorientation

        return space

    # TODO
    def _calculateRewards(self, data):
        return 0, False