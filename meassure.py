from time import sleep
import time
import robot
import math

arlo = robot.Robot()

leftSpeed = 66
rightSpeed = 63
waitTime = 0.042

ERR_MAR = 100


def find_edge(m: list):
    edge = 0.0
    for i in range(1, len(m)):
        if not (m[i] in range(m[i - 1] - ERR_MAR, m[i - 1] + ERR_MAR)):
            edge = m[i - 1]
            break
        if i == len(m) - 1:
            print("Can't find the edge")
    return edge


# Dist: negative for less than a meter positive for more than a meter
def drive_dist(dist: float):
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(2.3 - waitTime + dist)
    # print(arlo.go_diff(leftSpeed/2, 0, 1, 1))
    sleep(waitTime)
    arlo.stop()


def go_right_m():
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    m = []
    mid_dist = arlo.read_front_ping_sensor()
    end = time.time() + 0.7
    while time.time() < end:
        m.append(arlo.read_front_ping_sensor())
    print(arlo.stop())
    edge = find_edge(m)
    return [mid_dist, edge]


def go_left_m():
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    m = []
    mid_dist = arlo.read_front_ping_sensor()
    end = time.time() + 0.75
    while time.time() < end:
        m.append(arlo.read_front_ping_sensor())
    print(arlo.stop())
    edge = find_edge(m)
    return [mid_dist, edge]


print(go_right_m())
