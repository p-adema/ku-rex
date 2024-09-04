import time

import robot

arlo = robot.Robot()

for _ in range(10):
    print("Front sensor = ", arlo.read_front_ping_sensor())
    time.sleep(0.041)
