import cv2
import numpy as np

marker_size_mm = 144
avg_focal = 1105.6

arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters()
arucoParams.maxMarkerPerimeterRate = 10.0

_cam_internal = np.array(
    [
        [avg_focal, 0, 0],
        [0, avg_focal, 0],
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


class MarkerNotFound(ValueError):
    pass


def detect_markers(img: np.array):
    corners, ids, _ = cv2.aruco.detectMarkers(img, arucoDict, parameters=arucoParams)
    return corners, ids


def estimate_pose(cn, cam_internal=_cam_internal, obj_points=_object_points):
    ret, r_vec, t_vec = cv2.solvePnP(
        obj_points,
        imagePoints=cn,
        cameraMatrix=cam_internal,
        distCoeffs=_dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    if not ret:
        raise MarkerNotFound("Failed to solve PnP for pose estimation!")
    r_mat = np.empty((3, 3))
    cv2.Rodrigues(r_vec, dst=r_mat)
    return r_mat, t_vec
