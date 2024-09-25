from __future__ import annotations

import socket
import struct
import sys
import threading

import numpy as np

import aruco_utils

is_moving = False


def state_thread():
    global is_moving
    cam = aruco_utils.get_camera_picamera(downscale=1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("192.168.103.213", 1808))

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
            print(is_moving)
            markers = sample_markers()
            if not markers:
                s.send((b"m" if is_moving else b"s") + b"!empty")
                continue

            for name, _, t_vec in markers:
                msg.append(struct.pack("<Bdd", name, t_vec[0, 0], t_vec[2, 0]))
            send = b"".join(msg) + (b"m" if is_moving else b"s") + b"!end"
            s.send(send)
            msg.clear()
    except KeyboardInterrupt:
        pass
    finally:
        s.send(b"!close")
        s.close()


if __name__ == "__main__":
    t = threading.Thread(target=state_thread, args=[])
    t.start()

    import robot

    arlo = robot.Robot()
    input("Press enter to move")
    is_moving = True
    arlo.go(66, 64)

    t.join(timeout=1)

    arlo.stop()
    is_moving = False
    input("Press enter to exit.")
    sys.exit(0)

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
