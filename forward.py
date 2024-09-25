from time import sleep

import robot

# Create a robot object and initialize
arlo = robot.Robot()

input("Press enter")
# send a go_diff command to drive forward
arlo.go(66, 64)

# Wait a bit while robot moves forward
sleep(1)

# send a stop command
arlo.stop()

print("Finished")
