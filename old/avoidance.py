from collections import deque
from time import sleep, perf_counter


import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
left_speed = 40
right_speed = 40

waitTime = 0.041


def distanceStop():
    front = arlo.read_front_ping_sensor() < 300
    return front


def distanceGo():
    front = arlo.read_front_ping_sensor() > 400
    return front


def swerveRight(left, right):
    arlo.go_diff(left + 5, right, True, True)
    sleep(waitTime)
    while arlo.read_right_ping_sensor() < 50:
        pass
    arlo.go_diff(left, right, True, True)


def swerveLeft(left, right):
    arlo.go_diff(left, right + 10, True, True)
    sleep(waitTime)
    while arlo.read_right_ping_sensor() < 50:
        pass
    arlo.go_diff(left, right, True, True)


def drive(left, right):
    arlo.go(left, right)
    while not distanceStop():
        if arlo.read_right_ping_sensor() < 60:
            swerveLeft(left, right)
        if arlo.read_left_ping_sensor() < 60:
            swerveRight(left, right)
    arlo.stop()
    sleep(waitTime)
    avoidObstacle(left, right)


def courseReturn(left, right, i):
    arlo.go_diff(left, right, True, True)
    measurements = deque([arlo.read_right_ping_sensor() for _ in range(5)], maxlen=5)
    print("BEFORE")
    stop_time = perf_counter() + 2
    while perf_counter() < stop_time:
        measurements.append(arlo.read_right_ping_sensor())
        if sorted(measurements)[2] < 300:
            break

    for _ in range(5):
        measurements.append(arlo.read_right_ping_sensor())

    print("MIDDLE")
    while True:
        measurements.append(arlo.read_right_ping_sensor())
        if sorted(measurements)[2] > 300:
            break

    print("DONE")
    sleep(2.5)
    arlo.stop()
    arlo.go_diff(left, right, True, False)
    sleep(1.1)
    arlo.go_diff(left, right, True, True)
    sleep(0.5 * i)
    arlo.go_diff(left, right, False, True)
    sleep(1.1)
    print(i)


def creepLeft(left, right, t):
    arlo.go_diff(left, right, False, True)
    sleep(1.1)
    arlo.stop()
    sleep(waitTime)
    arlo.go_diff(left, right, True, True)
    sleep(t)
    arlo.stop()
    sleep(waitTime)
    arlo.go_diff(left, right, True, False)
    sleep(1.15)
    arlo.stop()
    sleep(waitTime)


def avoidObstacle(left, right):
    i = 0
    while not distanceGo():
        creepLeft(left, right, 0.5)
        i += 1
    creepLeft(left, right, 1)
    courseReturn(left, right, i + 2)
    drive(left, right)


input()
drive(left_speed, right_speed)

print("Finished")
