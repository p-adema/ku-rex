from __future__ import annotations

import contextlib
import queue
import socket
import threading
import time

import aruco_utils
import robot
from aruco_utils import sample_markers
from box_types import MovementAction, dedup_camera
from constants import server_ip, server_port
from kalman_state_fixed import KalmanState

movement_actions: queue.Queue[MovementAction] = queue.Queue()
is_running = True


def state_thread():
    global movement_actions, is_running
    cam = aruco_utils.get_camera_picamera(downscale=1)
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    try:
        s.connect((server_ip, server_port))
        msg = []
        state = KalmanState()

        while is_running:
            print(f"{state.state()}")

            img = cam.capture_array()[::1, ::1]
            timestamp = time.time()
            with contextlib.suppress(queue.Empty):
                move = movement_actions.get(block=False)
                # state.update_movement(move)

            markers = sample_markers(img)
            if not markers:
                s.send(b"!empty!end")
                continue
            try:
                boxes = dedup_camera(markers)
                state.update_camera(boxes, timestamp=timestamp)
            except AssertionError as e:
                if e.args:
                    e.args = (e.args[0] + f" ({markers=})",) + e.args[1:]
                else:
                    e.args = (f"No args AssertionError ({markers=})",)
                raise

            for box in boxes:
                msg.append(box.pack())

            msg.append(b"!state")

            est_state = state.state()
            msg.append(est_state.robot.pack())

            for s_box in est_state.boxes:
                msg.append(s_box.pack())

            msg.append(b"!end")

            s.send(b"".join(msg))
            msg.clear()
        s.send(b"!close")
    except ConnectionError as e:
        print(f"Connection error: {e}")
    finally:
        s.close()


def main_thread():
    global movement_actions, is_running
    t = threading.Thread(target=state_thread, args=[])
    t.start()
    try:
        arlo = robot.Robot()
        input("Press enter to move")
        movement_actions.put(MovementAction.moving(450))
        arlo.go(66, 64)
        t.join(timeout=6)
        arlo.stop()
        movement_actions.put(MovementAction.stop())
        input("Press enter to exit.")
    except KeyboardInterrupt:
        print("Quitting...")
    finally:
        is_running = False
    t.join()


if __name__ == "__main__":
    main_thread()

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
