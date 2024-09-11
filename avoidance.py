from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
left_speed = 64
right_speed = 64


waitTime = 0.041
print(arlo.go_diff(left_speed, right_speed, True, True))



sleep(3)
# send a stop command
print(arlo.stop())

def drive(left, right):
    arlo.go(left, right)
    while True:
        if arlo.read_front_ping_sensor() < 300:
            break
        sleep(waitTime)
        if arlo.read_left_ping_sensor() < 300:
            break
        sleep(waitTime)
        if arlo.read_right_ping_sensor() < 300:
            break
        sleep(waitTime)
        pass
    sleep(waitTime)
    arlo.stop()
    sleep(waitTime)
    avoidObstacle(left, right)

def avoidObstacle(left, right):
    arlo.go_diff(left_speed, right_speed, True, False)
    sleep(1)
    arlo.stop()

drive(left_speed, right_speed)
print("Finished")
