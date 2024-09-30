import math
from time import sleep

import robot

arlo = robot.Robot()

ROBOT_SPEED = 400  # milimeters per second
ROBOT_ROTATION = 0.0078  # seconds per degree
leftSpeed = 66
rightSpeed = 63
waitTime = 0.041


# Imprecise below 200mm
def go_foward(dist):
    print(f"Move {dist=}")
    time = dist / ROBOT_SPEED
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(time)
    print(arlo.stop())


# Turning is imprecise below 20 and above 200
def turnRight(theta):
    time = ROBOT_ROTATION * theta
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    sleep(time)
    print(arlo.stop())


def turnLeft(theta):
    time = ROBOT_ROTATION * theta
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(time)
    print(arlo.stop())


def turn(radians: float):
    print(f"Turn {radians=}")
    angle = math.degrees(radians)
    if abs(angle) < 5:
        return
    if angle > 0:
        turnLeft(angle)
    else:
        turnRight(-angle)


def stop():
    arlo.stop()


"""while True:
    temp = int(input('Enter distance'))
    turnLeft(temp)
    temp = int(input('Enter distance'))
    turnRight(temp)"""
