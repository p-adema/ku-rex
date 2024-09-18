# This script shows how to open a camera the picamera2 module and grab frames
# and show these.
# Kim S. Pedersen, 2023

import time
from pprint import pprint

import cv2  # Import the OpenCV library

try:
    import picamera2

    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)


print("OpenCV version = " + cv2.__version__)

# Open a camera device for capturing
imageSize = (1280, 720)
FPS = 30
cam = picamera2.Picamera2()
frame_duration_limit = int(1 / FPS * 1000000)  # Microseconds
# Change configuration to set resolution, framerate
picam2_config = cam.create_video_configuration(
    {"size": imageSize, "format": "RGB888"},
    controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
    queue=False,
)
cam.configure(picam2_config)  # Not really necessary
cam.start(show_preview=False)

pprint(cam.camera_configuration())  # Print the camera configuration in use

time.sleep(1)  # wait for camera to set up


# Open a window
WIN_RF = "Example 1"
cv2.namedWindow(WIN_RF)
cv2.moveWindow(WIN_RF, 100, 100)


while cv2.waitKey(4) == -1:  # Wait for a key pressed event
    image = cam.capture_array("main")[::4, ::4]

    # Show frames
    cv2.imshow(WIN_RF, image)


# Finished successfully
