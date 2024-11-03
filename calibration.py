from threading import Event

import move_calibrated

# from aruco_utils import get_camera_picamera
from kalman_state_fixed import KalmanStateFixed

with move_calibrated.CalibratedRobot() as robot:
    input("Ready")
    # sys.exit(0)
    s = KalmanStateFixed(9)
    e = Event()
    # cam = get_camera_picamera()
    # print("A")
    # print(cam.capture_array().shape)
    # print("B")

    robot.turn_left(180)
    while False:
        print(robot.arlo.read_front_ping_sensor())
        # img = cam.capture_array()
        # print(dedup_camera(sample_markers(img)))
        # robot.go_forward(np.array([0, 0]), np.array([0, int(input("dist? "))]))
        # robot.spin_left(state=s, event=e)
        # input()
        # robot.turn_left(45, stop=False)
        # robot.turn_left(360)
        # time.sleep(0.3)
