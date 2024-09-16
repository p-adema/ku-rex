import os
import time

import numpy as np

os.environ["OPENCV_LOG_LEVEL"] = "E"
import cv2
import picamera2

import aruco_utils


def get_camera_cv(
    capture_width: int = 1024, capture_height: int = 720, framerate: int = 30
) -> cv2.VideoCapture:
    """Utility function for getting a camera setup"""
    cam = cv2.VideoCapture(
        (
            "libcamerasrc !"
            "videobox autocrop=true !"
            "video/x-raw, "
            f"width=(int){capture_width}, "
            f"height=(int){capture_height}, "
            f"framerate=(fraction){framerate}/1 ! "
            "videoconvert ! "
            "appsink"
        ),
        apiPreference=cv2.CAP_GSTREAMER,
    )
    assert cam.isOpened(), "Could not open camera"
    return cam


def get_camera_picamera(fps: int = 30, image_size=(1280, 720)):
    cam = picamera2.Picamera2()
    frame_duration_limit = int(1 / fps * 1000000)  # Microseconds
    picam2_config = cam.create_video_configuration(
        {"size": image_size, "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
        queue=False,
    )
    cam.configure(picam2_config)
    cam.start(show_preview=False)
    return cam


cam = get_camera_picamera()
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
