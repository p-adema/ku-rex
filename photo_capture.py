import os

os.environ["OPENCV_LOG_LEVEL"] = "E"
import cv2
import numpy as np


def get_camera(
    capture_width: int = 1024, capture_height: int = 720, framerate: int = 30
) -> cv2.VideoCapture:
    """Utility function for getting a camera setup"""
    return cv2.VideoCapture(
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


# Define some constants
low_threshold = 35
ratio = 3
kernel_size = 3


# Open a camera device for capturing


def photo(name: str):
    cam = get_camera()
    assert cam.isOpened(), "Could not open camera"
    retval, image = cam.read()
    assert retval, "Failed to read image"
    np.save(f"data/photo_{name}.npy", image)
    cam.release()


try:
    while True:
        dist = int(input("Current distance: "))
        photo(f"approx_{dist}")
except KeyboardInterrupt:
    pass
