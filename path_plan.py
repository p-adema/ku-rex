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

stop_program = threading.Event()
scan_ready = threading.Event()
turning = threading.Event()

plan = None

def initial_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    robot: CalibratedRobot,
    known_boxes: list[Box],
    link: Link,
    start_t,
    turn_circle: bool = True
):  
    full_dur = robot.turn_right(360, ret_dur=True)
    print("scan")
    prev_t = start_t
    start_angle = state._angle
    print(start_angle)
    while True:
        img = cam.capture_array()
        new_t = time.time()
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
        dif = new_t - prev_t
        prev_t = new_t
        turned = 360*(dif/full_dur)
        print(dif, turned, full_dur)
        state.set_pos(turn=math.radians(-turned))
        link.send(boxes, state.current_state(), plan)

        if turning.isSet():
            img = cam.capture_array()
            timestamp = time.time()
            boxes = dedup_camera(sample_markers(img))
            state.update_camera(boxes, timestamp=timestamp)
            end_angle = state._angle
            l_turn = abs(end_angle - start_angle)
            print(start_angle, end_angle)
            print(l_turn)
            state.set_pos(turn=math.radians(-l_turn))
            link.send(boxes, state.current_state(), plan)
            print("break")
            break

    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img))
    print(f"{len(boxes)} found final scan")
    state.update_camera(boxes, timestamp=timestamp)
    if known_boxes:
        print("Pre-transform state", state.current_state())
        input("Press enter to transform")
        state.transform_known_boxes(known_boxes)
        print("Post-transform state", state.current_state())

    link.send(boxes, state.current_state(), plan)
    input("Press enter to continue")
    # print("Scan complete, state", state.current_state())


def state_thread(
    state: KalmanStateFixed,
    movement_actions: Queue[MovementAction],
    robot: CalibratedRobot,
):
    try:
        print("try")
        with Link(server_ip, server_port) as link:
            print("try success")
            cam = aruco_utils.get_camera_picamera()
            known_boxes = [
                Box(id=3, x=0, y=0),
                Box(id=6, x=-460, y=0),
                # Box(id=9, x=3_000, y=700),
            ]
            print("boxes, cam")
            initial_scan(cam, state, robot, known_boxes, link, time.time(), turn_circle=False)
            print("Initial Done")
            scan_ready.set()
            while not stop_program.is_set():
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
    movement_actions: Queue[MovementAction] = Queue()
    with CalibratedRobot() as robot:
        turning.clear()
        t = threading.Thread(target=state_thread, args=[state, movement_actions, robot])
        t.start()
        time.sleep(0.04)
        robot.turn_right(360)
        turning.set()
        current_node = 0
        scan_ready.wait()
        assert state.current_state().boxes, "No boxes found in scan! Can't plan!"
        while not stop_program.is_set():
            est_state = state.current_state()

            if plan is not None:
                # if np.linalg.norm(plan[current_node] - est_state.robot) < 100:
                current_node += 1
                if current_node == len(plan):
                    print("Achieved goal!")
                    break
                else:
                    angle, dist = state.propose_movement(
                        plan[current_node], pos=plan[current_node - 1]
                    )
                    robot.turn(angle)
                    print("Setting intermediate position")
                    state.set_pos(pos=plan[current_node - 1], turn=angle)
                    robot.go_forward(dist)
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
                    goal=Node(np.array([0, -500])),
                )[::-1]
                current_node = 1
                print("plan:", plan)
                angle, dist = state.propose_movement(plan[current_node], pos=plan[0])
                input("postplan.")
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
