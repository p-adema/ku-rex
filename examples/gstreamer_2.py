# This script shows how to do simple processing on frames coming from a camera in OpenCV
# Kim S. Pedersen, 2015

import cv2  # Import the OpenCV library
from pkg_resources import parse_version

OPCV3 = parse_version(cv2.__version__) >= parse_version("3")


def cap_prop_id(prop):
    """returns OpenCV VideoCapture property id given, e.g., "FPS
    This is needed because of differences in the Python interface in OpenCV 2.4 and 3.0
    """
    return getattr(
        cv2 if OPCV3 else cv2.cv, ("" if OPCV3 else "CV_") + "CAP_PROP_" + prop
    )


def gstreamer_pipeline(capture_width=1024, capture_height=720, framerate=30):
    """Utility function for setting parameters for the gstreamer camera pipeline"""
    return (
        "libcamerasrc !"
        "videobox autocrop=true !"
        "video/x-raw, "
        f"width=(int){capture_width:d}, "
        f"height=(int){capture_height:d}, "
        f"framerate=(fraction){framerate:d}/1 ! "
        "videoconvert ! "
        "appsink"
    )


print("OpenCV version = " + cv2.__version__)


# Define some constants
lowThreshold = 35
ratio = 3
kernel_size = 3


# Open a camera device for capturing
cam = cv2.VideoCapture(gstreamer_pipeline(), apiPreference=cv2.CAP_GSTREAMER)


if not cam.isOpened():  # Error
    print("Could not open camera")
    exit(-1)

# Get camera properties
width = int(cam.get(cap_prop_id("FRAME_WIDTH")))
height = int(cam.get(cap_prop_id("FRAME_HEIGHT")))
print("width = " + str(width) + ", Height = " + str(height))


# Open a window
WIN_RF = "Example 2"
cv2.namedWindow(WIN_RF)
cv2.moveWindow(WIN_RF, 100, 100)


# Preallocate memory
# gray_frame = np.zeros((height, width), dtype=np.uint8)

while cv2.waitKey(4) == -1:  # Wait for a key pressed event
    retval, frameReference = cam.read()  # Read frame

    if not retval:  # Error
        print(" < < <  Game over!  > > > ")
        exit(-1)

    # Convert the image to grayscale
    gray_frame = cv2.cvtColor(frameReference, cv2.COLOR_BGR2GRAY)

    # Reduce noise with a kernel 3x3
    edge_frame = cv2.blur(gray_frame, (3, 3))

    # Canny detector
    cv2.Canny(edge_frame, lowThreshold, lowThreshold * ratio, edge_frame, kernel_size)

    # Show frames
    cv2.imshow(WIN_RF, edge_frame)

# Close all windows
cv2.destroyAllWindows()

# Finished successfully
