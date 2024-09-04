import time

import robot

arlo = robot.Robot()

measurements = []
# arlo.go_diff(30, 30, True, False)
for _ in range(10):
    m = arlo.read_front_ping_sensor()
    # print("Front sensor = ", m)
    time.sleep(1 / 20)
    measurements.append(m)

print(measurements)
