from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()


sleep(1)


# send a go_diff command to drive forward
leftSpeed = 64
rightSpeed = 64
print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))

# Wait a bit while robot moves forward
sleep(3)

# send a stop command
print(arlo.stop())
sleep(0.05)

print("Left = " + str(arlo.read_left_wheel_encoder()))
sleep(0.05)
print("Right = " + str(arlo.read_right_wheel_encoder()))
sleep(0.05)


# send a go_diff command to drive backwards
leftSpeed = 64
rightSpeed = 64
print(arlo.go_diff(leftSpeed, rightSpeed, False, False))

# Wait a bit while robot moves forward
sleep(3)

# send a stop command
print(arlo.stop())
sleep(0.05)


print("Left = " + str(arlo.read_left_wheel_encoder()))
sleep(0.05)
print("Right = " + str(arlo.read_right_wheel_encoder()))
sleep(0.05)


# send a go_diff command to drive forward in a curve turning right
leftSpeed = 64
rightSpeed = 32
print(arlo.go_diff(leftSpeed, rightSpeed, True, True))

# Wait a bit while robot moves forward
sleep(3)

# send a stop command
print(arlo.stop())

sleep(0.05)

print("Left = " + str(arlo.read_left_wheel_encoder()))
sleep(0.05)
print("Right = " + str(arlo.read_right_wheel_encoder()))
sleep(0.05)
print(arlo.reset_encoder_counts())
sleep(0.05)
print("Left = " + str(arlo.read_left_wheel_encoder()))
sleep(0.05)
print("Right = " + str(arlo.read_right_wheel_encoder()))
sleep(0.05)

# send a go_diff command to drive backwards the same way we came from
print(arlo.go_diff(leftSpeed, rightSpeed, True, True))

# Wait a bit while robot moves backwards
sleep(3)

# send a stop command
print(arlo.stop())
sleep(0.05)

print("Left = " + str(arlo.read_left_wheel_encoder()))
sleep(0.05)
print("Right = " + str(arlo.read_right_wheel_encoder()))
sleep(0.05)


print("Finished")
