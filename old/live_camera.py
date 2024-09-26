import cv2  # Import the OpenCV library
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


print(f"{cv2.__version__=}")


# Define some constants
low_threshold = 35
ratio = 3
kernel_size = 3


# Open a camera device for capturing
cam = get_camera()


if not cam.isOpened():
    print("Could not open camera")
    exit(-1)


# Get camera properties
width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"{width =}, {height =}")


# Open a window
window_ref = "Example Window"
cv2.namedWindow(window_ref)
cv2.moveWindow(window_ref, 100, 100)


# Preallocate memory
image = np.empty((height, width), dtype=np.uint8)

cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)

while cv2.waitKey(4) == -1:  # Wait for a key pressed event
    retval, image = cam.read(image)
    assert retval, "Failed to read image"

    frame = cv2.cvtColor(image[::8, ::8], cv2.COLOR_BGR2GRAY)

    # Show frames
    cv2.imshow(window_ref, frame)

cv2.destroyAllWindows()
