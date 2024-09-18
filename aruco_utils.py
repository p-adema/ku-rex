import cv2
import numpy as np
import picamera2

marker_size_mm = 144
avg_focal = 1105.6

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.minDistanceToBorder = 0

_cam_internal = np.array(
    [
        [avg_focal, 0, 1280 // 2],
        [0, avg_focal, 720 // 2],
        [0, 0, 1],
    ]
)
_object_points = np.array(
    [
        [-marker_size_mm / 2, marker_size_mm / 2, 0],
        [marker_size_mm / 2, marker_size_mm / 2, 0],
        [marker_size_mm / 2, -marker_size_mm / 2, 0],
        [-marker_size_mm / 2, -marker_size_mm / 2, 0],
    ]
)
_dist_coeffs = np.zeros((4,))


def detect_markers(img: np.array) -> tuple[np.ndarray, np.ndarray]:
    # noinspection PyTypeChecker
    return cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)[:2]


def estimate_pose(cn, cam_internal=_cam_internal, obj_points=_object_points):
    ok, r_vec, t_vec = cv2.solvePnP(
        obj_points,
        imagePoints=cn,
        cameraMatrix=cam_internal,
        distCoeffs=_dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    assert ok, "Failed to solve PnP for pose estimation!"
    # r_mat = np.empty((3, 3))
    # cv2.Rodrigues(r_vec, dst=r_mat)
    return r_vec, t_vec


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


def get_camera_picamera(fps: int = 30, image_size=(1280, 720), downscale: int = 1):
    cam = picamera2.Picamera2()
    frame_duration_limit = int(1 / fps * 1000000)  # Microseconds
    image_size = image_size[0] // downscale, image_size[1] // downscale
    picam2_config = cam.create_video_configuration(
        {"size": image_size, "format": "RGB888"},
        controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
        buffer_count=1,
        queue=False,
    )
    cam.align_configuration(picam2_config)
    print("Camera size:", picam2_config["main"]["size"])
    cam.start(picam2_config)
    return cam


def calc_euler_angles(r_vec: np.ndarray) -> tuple[float, float, float]:
    r_mat = np.empty((3, 3))
    cv2.Rodrigues(r_vec, dst=r_mat)
    return (
        np.arctan2(r_mat[2, 1], r_mat[2, 2]),
        np.arctan2(-r_mat[2, 0], np.sqrt(r_mat[2, 1] ** 2 + r_mat[2, 2] ** 2)),
        np.arctan2(r_mat[1, 0], r_mat[0, 0]),
    )


def calc_turn_angle(t_vec: np.ndarray) -> float:
    return np.sign(t_vec[0]) * np.arcsin(abs(t_vec[0]) / t_vec[2]) / 0.44
