from __future__ import annotations

import socket
import threading

import aruco_utils
import robot
from aruco_utils import sample_markers, server_ip, server_port
from box_types import dedup_camera
from kalman_state import KalmanState

is_moving = False
is_running = True


def state_thread():
    global is_moving, is_running
    cam = aruco_utils.get_camera_picamera(downscale=1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect((server_ip, server_port))
        msg = []
        state = KalmanState()

        while is_running:
            print(f"{is_moving=}")
            img = cam.capture_array()[::1, ::1]
            markers = sample_markers(img)
            if not markers:
                s.send(b"!empty!end")
                continue

            boxes = dedup_camera(markers)
            state.update_camera(boxes)

            for box in boxes:
                msg.append(box.pack())

            msg.append(b"!state")

            for s_box in state.state().boxes:
                msg.append(s_box.pack())

            msg.append(b"!end")

            s.send(b"".join(msg))
            msg.clear()
    except KeyboardInterrupt:
        pass
    finally:
        s.send(b"!close")
        s.close()


def main_thread():
    global is_moving, is_running
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
    is_running = False


if __name__ == "__main__":
    main_thread()

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
