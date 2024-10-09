import move_calibrated

with move_calibrated.CalibratedRobot() as robot:
    input("Ready")
    robot.go_forward(500)
