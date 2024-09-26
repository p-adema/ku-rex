from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()


sleep(1)


# request to read Front sonar ping sensor
print("Front sensor = ", arlo.read_front_ping_sensor())

# request to read Back sonar ping sensor
print("Back sensor = ", arlo.read_back_ping_sensor())

# request to read Right sonar ping sensor
print("Right sensor = ", arlo.read_right_ping_sensor())

# request to read Left sonar ping sensor
print("Left sensor = ", arlo.read_left_ping_sensor())


# send a go_diff command to drive forward in a curve turning right
leftSpeed = 64
rightSpeed = 64
print(arlo.go_diff(leftSpeed, rightSpeed, True, True))

# Wait a bit while robot moves forward
sleep(3)


# send a go_diff command to drive forward in a curve turning right
leftSpeed = 64
rightSpeed = 64
print(arlo.go_diff(leftSpeed, rightSpeed, False, False))

# Wait a bit while robot moves forward
sleep(3)


# send a go_diff command to drive forward in a curve turning right
leftSpeed = 64
rightSpeed = 32
print(arlo.go_diff(leftSpeed, rightSpeed, True, True))

# Wait a bit while robot moves forward
sleep(3)

# send a stop command
print(arlo.stop())

sleep(0.05)


# send a go_diff command to drive backwards the same way we came from
print(arlo.go_diff(leftSpeed, rightSpeed, False, False))

# Wait a bit while robot moves backwards
sleep(3)

# send a stop command
print(arlo.stop())


print("Finished")
