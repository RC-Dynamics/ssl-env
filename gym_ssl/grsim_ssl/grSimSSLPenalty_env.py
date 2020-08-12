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
        Num     Observation                     Min                     Max
        0       Ball X   (cm)                   -700                    700
        1       Ball Y   (cm)                   -600                    600
        2       Ball Vx  (cm/s)                 -1000                   1000
        3       Ball Vy  (cm/s)                 -1000                   1000
        4       id 0 Blue Robot Y       (cm)    -600                    600
        5       id 0 Blue Robot Vy      (cm/s)  -1000                   1000
        6       id 0 Yellow Robot X     (cm)    -700                    700
        7       id 0 Yellow Robot Y     (cm)    -600                    600
        8       id 0 Yellow Robot Vx    (cm/s)  -1000                   1000
        9       id 0 Yellow Robot Vy    (cm/s)  -1000                   1000
        10      id 0 Yellow Robot Angle (rad)   -math.pi                math.pi

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
        obsSpaceThresholds = np.array([700, 600, 1000, 1000, 600, 1000, 700, 600, 100,
                                       math.pi], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obsSpaceThresholds, high=obsSpaceThresholds)

        print('Environment initialized')

    def step(self, actions):
        # Generate command Packet from actions
        packet = self._generateCommandPacket(actions)
        # Send command Packet
        self.client.send(packet)

        data = self.client.receive()
        while 0 not in [robot.robot_id for robot in data.detection.robots_blue]:
            data = self.client.receive()

        for robot in data.detection.robots_blue:
            if robot.robot_id == 0:
                observation = np.array([robot.x, robot.y, robot.orientation], dtype=np.float32)
        return observation, 0, False, {}

    def reset(self):
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
