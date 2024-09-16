from __future__ import annotations

import numpy as np

import aruco_utils

cam = aruco_utils.get_camera_picamera()
ts = []


def sample_distance(
    pixel_stride: int = 1,
) -> tuple[np.array | None, np.array | None]:
    img = cam.capture_array("main")[::pixel_stride, ::pixel_stride]
    corners, ids = aruco_utils.detect_markers(img)
    if ids is not None:
        try:
            return aruco_utils.estimate_pose(corners[0])
        except aruco_utils.MarkerNotFound:
            print("Failed to estimate pose!")
            return None, None
    else:
        return None, None


while True:
    r_vec, t_vec = sample_distance()
    if r_vec is not None:
        # print(
        #     str(np.round(r_vec.flatten(), 3)).ljust(40), np.round(t_vec.flatten() / 10)
        # )
        print(aruco_utils.calc_euler_angles(r_vec))

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
