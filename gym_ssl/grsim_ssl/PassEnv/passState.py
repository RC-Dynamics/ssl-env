from dataclasses import dataclass
from gym_ssl.grsim_ssl.Utils import *


ROBOT_RADIUS = 90

@dataclass
class passState:
  """Init Frame object."""
  ball_x: float = None
  ballY: float = None
  ball_vx: float = None
  ball_vy: float = None
  robot_vw: float = None
  dist_ra: float = None
  dist_rb: float = None
  dist_ab: float = None
  ally_x: float = None
  ally_y: float = None
  ally_sin: float = None
  ally_cos: float = None
  dist_ab: float = None
  real_ball_vx: float = None
  real_ball_vy: float = None
  timestamp: float = None

  def getRobotsDistance(self, frame) -> float:
    return float(mod(abs(frame.robotsBlue[0].x-frame.robotsYellow[0].x), abs(frame.robotsBlue[0].y-frame.robotsYellow[0].y)))

  def getAllyBallDistance(self, frame) -> float:
    return float(mod(abs(frame.ball.x-frame.robotsYellow[0].x), abs(frame.ball.y-frame.robotsYellow[0].y)))

  def getRobotBallDistance(self, frame) -> float:
    return float(mod(abs(frame.ball.x-frame.robotsBlue[0].x), abs(frame.ball.y-frame.robotsBlue[0].y)))

  def getRobotsPoleAngle(self, frame):
    dist_left = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - frame.robotsYellow[0].y]
    angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].theta))
    return math.sin(angle_left), math.cos(angle_left)

  def getAllyAngle(self, frame):
    dist_left = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - (frame.robotsYellow[0].y - ROBOT_RADIUS)]
    angle_left = toPiRange(angle(dist_left[0], dist_left[1]) + (math.pi - frame.robotsBlue[0].theta))
    return math.sin(angle_left), math.cos(angle_left)

  
  def getBallLocalCoordinates(self, frame):
    robot_ball = [frame.robotsBlue[0].x - frame.ball.x, frame.robotsBlue[0].y - frame.ball.y]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
    robot_ball_x = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_y = mod_to_ball* math.sin(angle_to_ball)
    
    return robot_ball_x, robot_ball_y

  def getAllyLocalCoordinates(self, frame):
    robot_ally = [frame.robotsBlue[0].x - frame.robotsYellow[0].x, frame.robotsBlue[0].y - frame.robotsYellow[0].y]
    mod_to_ally = mod(robot_ally[0], robot_ally[1])
    angle_to_ally = toPiRange(angle(robot_ally[0], robot_ally[1]) + (math.pi - frame.robotsBlue[0].theta))
    robot_ally_x = mod_to_ally* math.cos(angle_to_ally)
    robot_ally_y = mod_to_ally* math.sin(angle_to_ally)
    return robot_ally_x, robot_ally_y
  
  def getBallLocalSpeed(self, frame):
    robot_ball = [frame.robotsBlue[0].vx - frame.ball.vx, frame.robotsBlue[0].vy - frame.ball.vy]
    mod_to_ball = mod(robot_ball[0], robot_ball[1])
    angle_to_ball = toPiRange(angle(robot_ball[0], robot_ball[1]) + (math.pi - frame.robotsBlue[0].theta))
    robot_ball_vx = mod_to_ball* math.cos(angle_to_ball)
    robot_ball_vy = mod_to_ball* math.sin(angle_to_ball)
    return robot_ball_vx, robot_ball_vy

  
  def getObservation(self, frame):
    self.timestamp = (frame.timestamp)

    self.ball_x, self.ballY = self.getBallLocalCoordinates(frame)
    self.ball_vx, self.ball_vy = self.getBallLocalSpeed(frame)
    self.real_ball_vx, self.real_ball_vy = frame.ball.vx, frame.ball.vy

    self.ally_x, self.ally_y = self.getAllyLocalCoordinates(frame)
    self.ally_sin, self.ally_cos = self.getAllyAngle(frame)
    
    self.dist_ra = self.getRobotsDistance(frame)
    self.dist_rb  = self.getRobotBallDistance(frame)
    self.dist_ab  = self.getAllyBallDistance(frame)
    self.robot_vw = frame.robotsBlue[0].vw
    
    observation = []

    observation.append(self.ball_x) 
    observation.append(self.ballY) 
    observation.append(self.ball_vx) 
    observation.append(self.ball_vy)
    
    observation.append(self.ally_x)
    observation.append(self.ally_y)
    observation.append(self.ally_sin)
    observation.append(self.ally_cos)
    observation.append(self.dist_ab)
    observation.append(self.robot_vw)

    return observation