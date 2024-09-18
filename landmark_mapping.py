from __future__ import annotations

import socket
import struct

import numpy as np

import aruco_utils

cam = aruco_utils.get_camera_picamera(downscale=1)
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("192.168.128.213", 1807))
s.send(b"Hello")


def sample_markers(
    pixel_stride: int = 1,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    img = cam.capture_array()[::pixel_stride, ::pixel_stride]
    corners, ids = aruco_utils.detect_markers(img)
    res = []
    if ids is not None:
        for cn, i in zip(corners, ids):
            r_vec, t_vec = aruco_utils.estimate_pose(cn)
            res.append((int(i[0]), r_vec, t_vec))
    return res


try:
    msg = []

    while True:
        markers = sample_markers()
        if not markers:
            continue

        for name, _, t_vec in markers:
            msg.append(struct.pack("<Bdd", name, t_vec[0, 0], t_vec[2, 0]))
        send = b"".join(msg)
        s.send(send)
        msg.clear()

except KeyboardInterrupt:
    pass
finally:
    s.send(b"!close")
    s.close()
# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
