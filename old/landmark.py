import time

import numpy as np

import aruco_utils


def try_see(image):
    corners, ids = aruco_utils.detect_markers(image)
    if ids is not None:
        # print(len(ids), "markers detected", end=" ")
        r_vec, t_vec = aruco_utils.estimate_pose(corners[0])
        # total_dist = np.sqrt(np.square(t_vec).sum())
        # print(f"estimated distance: {total_dist:.0f}mm")
        print(
            str(np.round(r_vec.flatten(), 3)).ljust(40),
            np.round(t_vec.flatten() / 10),
            aruco_utils.calc_turn_angle(t_vec),
        )
    else:
        print("Can't see marker.")


cam = aruco_utils.get_camera_picamera()
ts = []
while True:
    start = time.perf_counter()
    # ok, img = cam.read()
    # assert ok, "Failed to read image"
    img = cam.capture_array("main")[::, ::]
    try_see(img)
    # zoom_size = 2
    # pad_width = (
    #     (img.shape[0] * (zoom_size // 2), img.shape[0] * (zoom_size // 2)),
    #     (img.shape[1] * (zoom_size // 2), img.shape[1] * (zoom_size // 2)),
    #     (0, 0),
    # )
    # pad_img = np.pad(img, ((0, 0), (0, 20), (0, 0)), mode="edge")
    # print("Main:")
    # try_see(img)
    # print("Padded:")
    # try_see(pad_img)
    # print("\n")
    ts.append(time.perf_counter() - start)

# print(min(ts), np.mean(ts), max(ts))
