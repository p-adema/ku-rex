import sys

import numpy as np

import aruco_utils
import box_types
import constants
from aruco_utils import get_camera_picamera

# Define some constants
low_threshold = 35
ratio = 3
kernel_size = 3


# Open a camera device for capturing

cam = get_camera_picamera()


def photo(name: str):
    # _ = cam.capture_buffer()
    image = cam.capture_buffer().reshape((constants.img_height, constants.img_width, 3))

    print(box_types.dedup_camera(aruco_utils.sample_markers(image)))
    np.save("data/chessboard.npy", image)
    sys.exit(0)


try:
    while True:
        # dist = int(input("Current distance: "))
        photo("why")
        # photo(f"new_foc_{dist}")
except KeyboardInterrupt:
    pass
