import time

import numpy as np

import aruco_utils

cam = aruco_utils.get_camera_picamera()
ts = []
while True:
    start = time.perf_counter()
    # ok, img = cam.read()
    # assert ok, "Failed to read image"
    img = cam.capture_array("main")[::, ::]
    corners, ids = aruco_utils.detect_markers(img)
    if ids is not None:
        # print(len(ids), "markers detected", end=" ")
        try:
            r_mat, t_vec = aruco_utils.estimate_pose(corners[0])
            total_dist = np.sqrt(np.square(t_vec).sum())
            print(f"estimated distance: {total_dist:.0f}mm")
        except aruco_utils.MarkerNotFound:
            print("Failed to estimate pose!")
    else:
        print("Can't see marker.")
    ts.append(time.perf_counter() - start)

# print(min(ts), np.mean(ts), max(ts))
