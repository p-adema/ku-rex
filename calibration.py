import move_calibrated

with move_calibrated.CalibratedRobot() as robot:
    input("Ready")
    robot.turn_left(180)
