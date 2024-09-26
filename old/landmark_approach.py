from __future__ import annotations

import time

import numpy as np

import aruco_utils
import robot

cam = aruco_utils.get_camera_picamera(downscale=1)
arlo = robot.Robot()
input("Press enter to start... ")


def sample_distance(
    pixel_stride: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    img = cam.capture_array()[::pixel_stride, ::pixel_stride]
    corners, ids = aruco_utils.detect_markers(img)
    if ids is not None:
        try:
            return aruco_utils.estimate_pose(corners[0])
        except aruco_utils.MarkerNotFound:
            print("Failed to estimate pose!")
            return None, None
    else:
        return None, None


spotted_grace = 0
approached_sonar = False
last_sonar = 1400
while True:
    if not spotted_grace:
        arlo.go(+44, -42, t=0.2)
        arlo.stop()
        time.sleep(0.4)

    r_vec, t_vec = sample_distance()

    if r_vec is not None:
        turn = aruco_utils.calc_turn_angle(t_vec)
        if turn > 0.3:
            arlo.go(+44, -42, t=0.05)
            arlo.stop()
        elif turn < -0.3:
            arlo.go(-44, +42, t=0.05)
            arlo.stop()
        else:
            left_correction = int(10 * turn)
            right_correction = int(-10 * turn)
            dist = t_vec[2]
            sonar = arlo.read_front_ping_sensor()
            print(sonar)
            if dist > 550:
                arlo.go(+66 + left_correction, +64 + right_correction, t=0.2)
                last_sonar = 1400
                approached_sonar = False
            elif 200 < sonar < last_sonar:
                print("sonar!", sonar)
                last_sonar = sonar
                arlo.go(+66 + left_correction, +64 + right_correction, t=0.1)
                approached_sonar = True
            else:
                print("Close!")
                arlo.stop()
        spotted_grace = 3
    elif spotted_grace:
        if approached_sonar:
            sonar = arlo.read_front_ping_sensor()
            if 300 < sonar < last_sonar:
                last_sonar = sonar
                arlo.go(+66, +64, t=0.1)
            elif sonar < 300:
                print("Sonar close!")
                arlo.stop()
            else:
                spotted_grace -= 1
                last_sonar = 1400
                approached_sonar = False
        else:
            print("Lost sight!")
            spotted_grace -= 1
