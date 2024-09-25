import matplotlib.pyplot as plt
import numpy as np
import aruco_utils
import math

ROBOT_DIA = 55
ROBOT_R = ROBOT_DIA / 2
ROBOT_CORD = (0, -22.5, 0)

cam = aruco_utils.get_camera_picamera(downscale=1)

"""
Object detection

returns a tuple of 3 elements or an object
"""


def find_objs(img) -> tuple[int, int, int]:
    objs = []
    corners, ids = aruco_utils.detect_markers(img)
    if ids is not None:
        for cn, i in zip(corners, ids):
            r_vecs, t_vecs = aruco_utils.estimate_pose(cn)
            pos = t_vecs / 10
            objs.append((int(i[0]), int((pos[0])[0]), int((pos[2])[0])))
    return objs


""" Coalition detection between the robot and an object"""


def col_detect(oi, ri=50) -> bool:
    o = ROBOT_CORD
    xd = (o[1] - oi[1]) ** 2
    yd = (o[2] - oi[2]) ** 2
    d = math.sqrt(xd + yd)
    return d <= ROBOT_R + ri
