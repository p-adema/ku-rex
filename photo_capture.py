import os

os.environ["OPENCV_LOG_LEVEL"] = "E"
import numpy as np

from aruco_utils import get_camera_picamera

# Define some constants
low_threshold = 35
ratio = 3
kernel_size = 3


# Open a camera device for capturing


def photo(name: str):
    cam = get_camera_picamera()
    image = cam.capture_buffer()
    np.save(f"data/photo_{name}.npy", image)
    cam.close()


try:
    while True:
        dist = int(input("Current distance: "))
        photo(f"focal_{dist}")
except KeyboardInterrupt:
    pass
