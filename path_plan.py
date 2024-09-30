from __future__ import annotations

import contextlib
import queue
import socket
import struct
import threading
import time

import numpy as np

import aruco_utils
import move_calibrated
from aruco_utils import sample_markers
from box_types import Box, MovementAction, StateEstimate, dedup_camera
from constants import server_ip, server_port
from kalman_state_fixed import KalmanStateFixed
from rrt_landmarks import RRT, Node

movement_actions: queue.Queue[MovementAction] = queue.Queue()
is_running = True
state = KalmanStateFixed()
plan = None


class Link:
    def __init__(self, ip, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip, port))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.s.close()

    def empty(self):
        self.s.send(b"!empty!end")

    def send(self, boxes: list[Box], est_state: StateEstimate):
        msg = []
        for box in boxes:
            msg.append(box.pack())
        msg.append(b"!state")

        msg.append(est_state.robot.pack())

        for s_box in est_state.boxes:
            msg.append(s_box.pack())

        msg.append(b"!path")

        if plan is not None:
            path_flat = plan.flatten()
            msg.append(struct.pack(f"<{len(path_flat)}d", *path_flat))

        msg.append(b"!end")
        self.s.send(b"".join(msg))
        msg.clear()


def state_thread():
    global movement_actions, is_running, state
    cam = aruco_utils.get_camera_picamera(downscale=1)

    with Link(server_ip, server_port) as link:
        while is_running:
            print(f"{state.current_state()}")
            img = cam.capture_array()[::1, ::1]
            timestamp = time.time()
            with contextlib.suppress(queue.Empty):
                move = movement_actions.get(block=False)
                state.update_movement(move)

            markers = sample_markers(img)
            if not markers:
                link.empty()
                continue
            boxes = dedup_camera(markers)
            state.update_camera(boxes, timestamp=timestamp)

            est_state = state.current_state()
            link.send(boxes, est_state)


def main_thread():
    global movement_actions, is_running, plan
    t = threading.Thread(target=state_thread, args=[])
    t.start()
    try:
        current_node = 0
        while is_running:
            est_state = state.current_state()
            if plan is None and not est_state.boxes:
                continue

            if plan is not None:
                # if np.linalg.norm(plan[current_node] - est_state.robot) < 100:
                current_node += 1
                if current_node == len(plan):
                    print("Achieved goal!")
                    is_running = False
                else:
                    angle, dist = state.propose_movement(
                        plan[current_node], pos=plan[current_node - 1]
                    )
                    move_calibrated.turn(angle)
                    state._angle += angle
                    move_calibrated.go_foward(dist)
                # else:
                #     print(
                #         f"Far away :( {est_state.robot=} {plan[current_node]=}",
                #     )
                #     pass
            else:
                movement_actions.put(MovementAction.moving(1))
                plan = RRT.generate_plan(
                    landmarks=est_state.boxes,
                    start=est_state.robot,
                    goal=Node(
                        np.array([est_state.boxes[0].x, est_state.boxes[0].y + 900])
                    ),
                )[::-1]
                current_node = 1
                print("plan:", plan)
                angle, dist = state.propose_movement(plan[current_node], pos=plan[0])
                move_calibrated.turn(angle)
                state._angle += angle
                print(f"New angle: {state._angle}")
                move_calibrated.go_foward(dist)
    except KeyboardInterrupt:
        print("Quitting...")
    finally:
        move_calibrated.stop()
        is_running = False
    t.join()


if __name__ == "__main__":
    main_thread()

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
