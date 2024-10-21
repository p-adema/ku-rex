from __future__ import annotations

import contextlib
import math
import threading
import time
from queue import Empty, Queue

import numpy as np
import picamera2

import aruco_utils
from aruco_utils import sample_markers
from box_types import Box, MovementAction, dedup_camera
from client_link import Link
from constants import server_ip, server_port
from kalman_state_fixed import KalmanStateFixed
from move_calibrated import CalibratedRobot
from rrt_landmarks import RRT, Node

KNOWN_BOXES = [
    Box(id=4, x=0, y=0),
    Box(id=1, x=1950, y=0),
    Box(id=8, x=820, y=1980),
]

GOAL = Node(np.array([820, 1000]))

stop_program = threading.Event()
scan_ready = threading.Event()
do_scan = threading.Event()

plan = None


def circular_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    robot: CalibratedRobot,
    known_boxes: list[Box],
    link: Link,
):
    for _ in range(18):
        print("Angle:", math.degrees(state._angle))
        img = cam.capture_array()
        timestamp = time.time()
        boxes = dedup_camera(sample_markers(img))
        state.update_camera(boxes, timestamp=timestamp)
        # print(f"{len(boxes)} found scan {i} ({boxes} -> {state.current_state()})")
        # print(
        #     "Position variance:",
        #     np.round(
        #         np.linalg.norm(np.diag(state._cur_covar).reshape((-1, 2)), axis=1)
        #     ),
        # )
        robot.scan_left()
        state.set_pos(turn=math.radians(20))
        link.send(boxes, state.current_state(), plan)
        time.sleep(0.4)

    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img))
    # print(f"{len(boxes)} found final scan")
    state.update_camera(boxes, timestamp=timestamp)
    if known_boxes:
        print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        state.transform_known_boxes(known_boxes)
        print("Post-transform state", state.current_state())

    link.send(boxes, state.current_state(), plan)
    # input("Press enter to continue")
    # print("Scan complete, state", state.current_state())


def state_thread(
    state: KalmanStateFixed,
    movement_actions: Queue[MovementAction],
    robot: CalibratedRobot,
):
    try:
        with Link(server_ip, server_port) as link:
            cam = aruco_utils.get_camera_picamera(downscale=1)
            while not stop_program.is_set():
                if do_scan.is_set():
                    do_scan.clear()
                    circular_scan(cam, state, robot, KNOWN_BOXES, link)
                    KNOWN_BOXES.clear()  # hacky way of doing it once
                    scan_ready.set()

                print(f"State thread: {state.current_state()}")
                img = cam.capture_array()
                timestamp = time.time()
                with contextlib.suppress(Empty):
                    move = movement_actions.get(block=False)
                    state.update_movement(move)

                markers = sample_markers(img)
                if not markers:
                    link.send([], state.current_state(), plan)
                    link.empty()
                    # state.advance(timestamp)
                    continue
                boxes = dedup_camera(markers)
                # state.update_camera(boxes, timestamp=timestamp)

                est_state = state.current_state()
                link.send(boxes, est_state, plan)
    finally:
        print("State exitting!")
        scan_ready.set()
        stop_program.set()


def main_thread():
    global plan
    state = KalmanStateFixed(n_boxes=9)
    movement_actions: Queue[MovementAction] = Queue()
    with CalibratedRobot() as robot:
        t = threading.Thread(target=state_thread, args=[state, movement_actions, robot])
        t.start()
        current_node = 0
        while not stop_program.is_set():
            do_scan.set()
            scan_ready.wait()
            scan_ready.clear()
            assert state.current_state().boxes, "No boxes found in scan! Can't plan!"
            est_state = state.current_state()
            if np.linalg.norm(np.asarray(est_state.robot) - np.asarray(GOAL)) < 100:
                print("Reached goal!")
                break

            movement_actions.put(MovementAction.moving(1))
            plan = RRT.generate_plan(
                landmarks=est_state.boxes,
                start=est_state.robot,
                goal=GOAL,
            )[::-1]
            current_node = 1
            print("plan:", plan)
            angle, dist = state.propose_movement(plan[current_node], pos=plan[0])
            # input("postplan.")
            state.set_pos(pos=plan[0], turn=angle)
            print("postplan-enter")

            robot.turn(angle)
            robot.go_forward(dist)
    stop_program.set()
    t.join()


if __name__ == "__main__":
    try:
        main_thread()
    finally:
        stop_program.set()
        print("Exiting...")

# left: -3 -0.02, -0.001
# right: 3, 0.05, -0.01

# Centre one is important:  positive means right side,  negative means left side
#                           straight on, approximately zero
