## State
from dataclasses import dataclass
from gym_ssl.grsim_ssl.Utils import *

CENTER_GOAL_X = -6000
CENTER_GOALY = 0

LEFT_GOAL_X = -6000
LEFT_GOALY = -600

RIGHT_GOAL_X = -6000
RIGHT_GOALY = 600

ROBOT_RADIUS = 90

@dataclass
class goalieState:
    """Init Frame object."""
    ball_x: float = None
    ballY: float = None
    ball_vx: float = None
    ball_vy: float = None
    robot_w: float = None
    distance: float = None
    theta_l_sen: float = None
    theta_l_cos: float = None
    theta_r_sen: float = None
    theta_r_cos: float = None
    theta_goalie_c_sen: float = None
    theta_goalie_c_cos: float = None
    theta_goalie_l_sen: float = None
    theta_goalie_l_cos: float = None
    theta_goalie_r_sen: float = None
    theta_goalie_r_cos: float = None

    def getDistance(self, frame) -> float:
        return float(mod(abs(frame.robotsBlue[0].x-CENTER_GOAL_X), abs(frame.robotsBlue[0].y-CENTER_GOALY)))

    def getLeftPoleAngle(self, frame):
        dist_left = [frame.robotsBlue[0].x - LEFT_GOAL_X, frame.robotsBlue[0].y - LEFT_GOALY]
        angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].theta))
        return math.sin(angle_left), math.cos(angle_left)

    def getRightPoleAngle(self, frame):
        dist_right = [frame.robotsBlue[0].x - RIGHT_GOAL_X, frame.robotsBlue[0].y - RIGHT_GOALY]
        angle_right = toPiRange(angle(dist_right[0], dist_right[1]) + (math.pi - frame.robotsBlue[0].theta))
        return math.sin(angle_right), math.cos(angle_right)

    def getGoalieCenterUnifiedAngle(self, frame):
        dist_g = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - frame.robotsYellow[0].y]
        angle_g = toPiRange(angle(dist_g[0], dist_g[1]) + (math.pi - frame.robotsBlue[0].theta))
        return angle_g

    def getGoalieCenterAngle(self, frame):
        angle_c = self.getGoalieCenterUnifiedAngle(frame)
        return math.sin(angle_c), math.cos(angle_c)

    def getGoalieLeftAngle(self, frame):
        dist_left = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].y - ROBOT_RADIUS)]
        angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].theta))
        return math.sin(angle_left), math.cos(angle_left)

    def getGoalieRightAngle(self, frame):
        dist_right = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].y + ROBOT_RADIUS)]
        angle_right = toPiRange(angle(dist_right[0], dist_right[1]) + (math.pi - frame.robotsBlue[0].theta))
        return math.sin(angle_right), math.cos(angle_right)

    def getBallLocalCoordinates(self, frame):
        robot_ball = [frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y]
        mod_to_ball = mod(robot_ball[0], robot_ball[1])
        angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
        robot_ball_x = mod_to_ball * math.cos(angle_to_ball)
        robot_ball_y = mod_to_ball * math.sin(angle_to_ball)
        return robot_ball_x, robot_ball_y

    def getBallLocalSpeed(self, frame):
        robot_ball = [frame.robotsBlue[0].vx - frame.ball.vx, frame.robotsBlue[0].vy - frame.ball.vy]
        mod_to_ball = mod(robot_ball[0], robot_ball[1])
        angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
        robot_ball_vx = mod_to_ball* math.cos(angle_to_ball)
        robot_ball_vy = mod_to_ball* math.sin(angle_to_ball)
        return robot_ball_vx, robot_ball_vy

    def getRelativeRobotToBallAngle(self, frame):
        
        robot_ball = [frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y]
        angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
        #print(angle_to_ball)
        return angle_to_ball

    def getRelativeRobotToBallDist(self, frame):
        
        robot_ball = [frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y]
        dist = (robot_ball[0]**2 + robot_ball[1]**2)**(1/2)
        #print(angle_to_ball)
        return dist
    
    def getGoalDist(self, frame):
        xdist_goal  = frame.robotsBlue[0].x - CENTER_GOAL_X
        ydistl_goal = frame.robotsBlue[0].y - LEFT_GOALY
        ydistr_goal = frame.robotsBlue[0].y - RIGHT_GOALY
        return xdist_goal, ydistl_goal, ydistr_goal
        

    def getObservation(self, frame):
        """
    Observation:
        Type: Box(13)
        Num     Observation                                       Min                     Max
        0       Rel Ball X   (mm)                               -7000                   7000
        1       Rel Ball Y   (mm)                               -6000                   6000
        2       Ball Vx  (mm/s)                                 -10000                  10000
        3       Ball Vy  (mm/s)                                 -10000                  10000
        6       Blue id 0 Vx  (mm/s)                            -10000                  10000
        7       Blue id 0 Vy  (mm/s)                            -10000                  10000
        8       Blue id 0 Robot Theta    (rad)                  -math.pi                math.pi 
        9       Blue id 0 Robot Vw       (rad/s)                -math.pi * 3            math.pi * 3
        10      Dist Blue id0 - ball (mm)                       -5000                   5000
        11      Angular Dist Blue id0 - ball (mm)               -math.pi                math.pi
        12      Dist X Blue id0 - goal centerX (mm)             -2000                   2000
        13      Dist Y Blue id0 - goal left (mm)                -2000                   2000 
        14      Dist Y Blue id0 - goal right (mm)               -2000                   2000
        """

        self.ball_x, self.ballY = self.getBallLocalCoordinates(frame)
        self.ball_vx, self.ball_vy = self.getBallLocalSpeed(frame)

        self.distance = self.getDistance(frame)
        self.robot_w = frame.robotsBlue[0].vw
        self.theta_l_sen, self.theta_l_cos = self.getLeftPoleAngle(frame)
        self.theta_r_sen, self.theta_r_cos = self.getRightPoleAngle(frame)
        self.theta_goalie_c_sen, self.theta_goalie_c_cos = self.getGoalieCenterAngle(frame)
        self.theta_goalie_l_sen, self.theta_l_cos = self.getGoalieLeftAngle(frame)
        self.theta_goalie_r_sen, self.theta_r_cos = self.getGoalieRightAngle(frame)
        self.angle_relative = self.getRelativeRobotToBallAngle(frame)
        self.robot_ball_dist = self.getRelativeRobotToBallDist(frame)
        xdist_goal, ydistl_goal, ydistr_goal = self.getGoalDist(frame)

        observation = []

        observation.append(self.ball_x)  #0
        observation.append(self.ballY)   #1
        observation.append(self.ball_vx) #2 
        observation.append(self.ball_vy) #3
        
        observation.append(frame.robotsBlue[0].vx ) #6
        observation.append(frame.robotsBlue[0].vy ) #7
        observation.append(frame.robotsBlue[0].theta) #8
        observation.append(frame.robotsBlue[0].vw)  #9     
        
        observation.append(self.robot_ball_dist)    #10
        observation.append(self.angle_relative)     #11
        
        observation.append(xdist_goal)  #12
        observation.append(ydistl_goal) #13
        observation.append(ydistr_goal)  #14

        return observation