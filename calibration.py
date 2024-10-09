import time

import move_calibrated

with move_calibrated.CalibratedRobot() as robot:
    input("Ready")
    # with robot.go_forward(np.array([0, 0]), np.array([0, 2_000])) as _:
    #     pass
    # sys.exit(0)
    for _ in range(1):
        with robot.turn_left(360):
            pass
        time.sleep(0.3)
