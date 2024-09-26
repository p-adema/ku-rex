import time

import numpy as np

import robot

arlo = robot.Robot()
input("Press enter to run...")
start = time.perf_counter()
arlo.go(-64, +62)
measurements = []
while time.perf_counter() - start < 3:
    measurements.append(arlo.sonar())

np.save("data/arc.npy", np.array(measurements, dtype=np.uint16))
arlo.stop()
