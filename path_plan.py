from __future__ import annotations

import threading
import time

import numpy as np
import picamera2

import aruco_utils
from aruco_utils import sample_markers
from box_types import Box, dedup_camera
from client_link import Link
from constants import server_ip, server_port
from kalman_state_fixed import KalmanStateFixed
from move_calibrated import CalibratedRobot
from rrt_landmarks import RRT, Node

GOAL = Node(np.array([820, 1500]))

KNOWN_BOXES = [
    Box(id=4, x=0, y=0),
    Box(id=1, x=1750, y=0),
    Box(id=8, x=820, y=1980),
]

stop_program = threading.Event()
turn_ready = threading.Event()
scan_ready = threading.Event()

plan = None


def initial_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    link: Link,
    do_update: bool = True,
):
    turn_ready.wait()
    turn_ready.clear()
    while not turn_ready.is_set():
        img = cam.capture_array()
        timestamp = time.time()
        boxes = dedup_camera(sample_markers(img))
        state.update_camera(boxes, timestamp=timestamp)
        link.send(boxes, state.current_state(), plan)
    turn_ready.clear()
    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img))
    print(f"{len(boxes)} found final scan")
    state.update_camera(boxes, timestamp=timestamp)
    if KNOWN_BOXES and do_update:
        print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        state.transform_known_boxes(KNOWN_BOXES)
        print("Post-transform state", state.current_state())

    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img))
    state.update_camera(boxes, timestamp=timestamp)

    link.send(boxes, state.current_state(), plan)
    # input("Press enter to continue")
    # print("Scan complete, state", state.current_state())


def state_thread(
    state: KalmanStateFixed,
):
    try:
        with Link(server_ip, server_port) as link:
            cam = aruco_utils.get_camera_picamera()
            print("Scan 1")
            initial_scan(cam, state, link, do_update=True)
            print("Scan 2")
            scan_ready.set()
            initial_scan(cam, state, link, do_update=False)
            scan_ready.set()
            print("Scan done!")
            while not stop_program.is_set():
                # print(f"State thread: {state.current_state()}")
                img = cam.capture_array()
                timestamp = time.time()

                markers = sample_markers(img)
                if not markers:
                    link.send([], state.current_state(), plan)
                    continue
                boxes = dedup_camera(markers)
                print(f"Found boxes! {timestamp}")
                # state.update_camera(boxes, timestamp=timestamp)

                est_state = state.current_state()
                link.send(boxes, est_state, plan)
    finally:
        print("State exiting!")
        scan_ready.set()
        stop_program.set()


def main_thread():
    global plan
    state = KalmanStateFixed(n_boxes=9)
    with CalibratedRobot() as robot:
        t = threading.Thread(target=state_thread, args=[state])
        t.start()
        current_node = 0
        for _ in range(2):
            timestamp = robot.prepare_left()
            turn_ready.set()
            robot.turn_left(360, state=state, actual_start=timestamp)
            turn_ready.set()
            scan_ready.wait()
            scan_ready.clear()
        assert state.current_state().boxes, "No boxes found in scan! Can't plan!"
        while not stop_program.is_set():
            est_state = state.current_state()

            if plan is None:
                plan_res = RRT.generate_plan(
                    landmarks=est_state.boxes,
                    start=est_state.robot,
                    goal=GOAL,
                    max_iter=2_000,
                )
                if plan_res is None:
                    print("Couldn't find a plan!")
                    break
                plan = plan_res[::-1]
                current_node = 1
                print("plan:", plan)
                angle, _dist = state.propose_movement(plan[current_node], pos=plan[0])
                input("postplan.")
                print("postplan-enter")

                robot.turn(angle, state=state)
                robot.go_forward(plan[0], plan[1], state=state)
                continue

            current_node += 1
            if current_node == len(plan):
                print("Achieved goal!")
                break

            angle, _dist = state.propose_movement(
                plan[current_node], pos=plan[current_node - 1]
            )
            robot.turn(angle, state=state)
            robot.go_forward(plan[current_node - 1], plan[current_node], state=state)

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
# t
