from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

import aruco_utils

cam = aruco_utils.get_camera_picamera(downscale=1)


def sample_markers(
    pixel_stride: int = 1,
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    img = cam.capture_array()[::pixel_stride, ::pixel_stride]
    corners, ids = aruco_utils.detect_markers(img)
    res = {}
    if ids is not None:
        for cn, i in zip(corners, ids):
            res[i[0]] = aruco_utils.estimate_pose(cn)

    return res


ax = plt.gca()

while True:
    markers = sample_markers()
    print(markers)
    for name, (_, t_vec) in markers.items():
        plt.scatter(t_vec[0], t_vec[2], label=name)
    break

plt.show()
# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
