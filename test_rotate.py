import time

import move_calibrated

with move_calibrated.CalibratedRobot() as robot:
    input("ready")
    for _ in range(18):
        robot.scan_left()
        time.sleep(1)
