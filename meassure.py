from time import sleep
import time
import robot
import math

arlo = robot.Robot()

leftSpeed = 66
rightSpeed = 63
waitTime = 0.042

ERR_MAR = 100

# Dist: negative for less than a meter positive for more than a meter
def drive(dist : float):
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(2.3 - waitTime + dist)
    # print(arlo.go_diff(leftSpeed/2, 0, 1, 1))
    sleep(waitTime)
    arlo.stop()

def go_right_m():
    angle = 55
    m = []
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    end = time.time() + 0.7
    while time.time() < end:
        left = arlo.read_left_ping_sensor()
        mid = arlo.read_front_ping_sensor()
        if left + ERR_MAR in range(mid-ERR_MAR, mid+ERR_MAR) or left - ERR_MAR in range(mid-ERR_MAR, mid+ERR_MAR):
            m.append(math.sqrt(left**2 + mid**2-2*left*mid*math.cos(angle)))
    print(arlo.stop())
    return m

# 0 = left, 1 = right
def pass_obstacle(dir : bool):
    dists = []

    if dir == True:
            calibration.turnRight()
            left = arlo.read_left_ping_sensor()
            mid = arlo.read_front_ping_sensor()
            right = arlo.read_right_ping_sensor()

    

print(go_right_m())