import cv2
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

    markers = aruco_utils.sample_markers(image)
    if markers:
        box = markers[0]
        rot = np.empty((3, 3))
        cv2.Rodrigues(box.r_vec, dst=rot)
        angle = cv2.RQDecomp3x3(rot)[0][1]
        print(f"Box {box.id} angle {angle}")
    print(box_types.dedup_camera(markers))
    # np.save("data/chessboard.npy", image)
    # sys.exit(0)


try:
    while True:
        # dist = int(input("Current distance: "))
        photo("why")
        # photo(f"new_foc_{dist}")
except KeyboardInterrupt:
    pass
