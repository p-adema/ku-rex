from __future__ import annotations

import socket
import struct
import threading

import aruco_utils
import robot
from box_types import CameraBox, dedup_camera
from kalman_state import KalmanState

is_moving = False
running = True


def sample_markers(
    cam,
    pixel_stride: int = 1,
) -> list[CameraBox]:
    img = cam.capture_array()[::pixel_stride, ::pixel_stride]
    corners, ids = aruco_utils.detect_markers(img)
    res = []
    if ids is not None:
        for cn, i in zip(corners, ids):
            r_vec, t_vec = aruco_utils.estimate_pose(cn)
            res.append(CameraBox(id=int(i[0]), r_vec=r_vec, t_vec=t_vec))

    return res


def state_thread():
    global is_moving, running
    cam = aruco_utils.get_camera_picamera(downscale=1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect(("192.168.103.213", 1808))
        msg = []
        state = KalmanState()

        while running:
            print(is_moving)
            markers = sample_markers(cam)
            if not markers:
                s.send((b"m" if is_moving else b"s") + b"!empty")
                continue

            boxes = dedup_camera(markers)
            state.update_camera(boxes)

            for box in boxes:
                msg.append(box.pack())

            msg.append(b'!state')

            for s_box in state.state()[1]:
                msg.append(s_box.pack())

            msg.append(b"!end")

            s.send(b"".join(msg))
            msg.clear()
    except KeyboardInterrupt:
        pass
    finally:
        s.send(b"!close")
        s.close()


def main():
    global is_moving, running
    t = threading.Thread(target=state_thread, args=[])
    t.start()
    arlo = robot.Robot()
    input("Press enter to move")
    is_moving = True
    arlo.go(66, 64)
    t.join(timeout=1)
    arlo.stop()
    is_moving = False
    input("Press enter to exit.")
    running = False


if __name__ == "__main__":
    main()

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
