from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()

print("Running ...")


# send a go_diff command to drive forward
left_speed = 64
right_speed = 64
print(arlo.go_diff(left_speed, right_speed, True, True))

# Wait a bit while robot moves forward
sleep(3)

# send a stop command
print(arlo.stop())

# Wait a bit before next command
sleep(0.041)

# send a go_diff command to drive backwards the same way we came from
print(arlo.go_diff(left_speed, right_speed, True, True))

# Wait a bit while robot moves backwards
sleep(3)

# send a stop command
print(arlo.stop())

# Wait a bit before next command
sleep(0.041)


# request to read Front sonar ping sensor
print("Front sensor = ", arlo.read_front_ping_sensor())
sleep(0.041)


# request to read Back sonar ping sensor
print("Back sensor = ", arlo.read_back_ping_sensor())
sleep(0.041)

# request to read Right sonar ping sensor
print("Right sensor = ", arlo.read_right_ping_sensor())
sleep(0.041)

# request to read Left sonar ping sensor
print("Left sensor = ", arlo.read_left_ping_sensor())
sleep(0.041)


# send a go_diff command to drive forward in a curve turning right
left_speed = 64
right_speed = 32
print(arlo.go_diff(left_speed, right_speed, True, True))

# Wait a bit while robot moves forward
sleep(3)

# send a stop command
print(arlo.stop())

# Wait a bit before next command
sleep(0.041)

# send a go_diff command to drive backwards the same way we came from
print(arlo.go_diff(left_speed, right_speed, False, False))

# Wait a bit while robot moves backwards
sleep(3)

# send a stop command
print(arlo.stop())


print("Finished")
