from __future__ import annotations

import math
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
from movement_predictors import Stopped
from rrt_landmarks import RRT, Node

AUTO_SCAN_INTERVAL = 15
GOAL_UPDATE_INTERVAL = 1

CURRENT_GOAL = Node(np.array([0, 0]))

KNOWN_BOXES = [
    Box(id=5, x=0, y=0),
    Box(id=3, x=1_290, y=940),
    Box(id=4, x=280, y=1_850),
]
N_START_SCANS = 1

stop_program = threading.Event()
turn_ready = threading.Event()
scan_ready = threading.Event()

turn_barrier = threading.Barrier(2)
re_scan_barrier = threading.Barrier(2)

plan = None


def circular_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    link: Link,
    do_update: bool = True,
) -> bool:
    state.force_box_uncertainty()
    turn_ready.wait()
    turn_ready.clear()
    while not turn_ready.is_set():
        img = cam.capture_array()
        timestamp = time.time()
        boxes = dedup_camera(sample_markers(img))
        state.update_camera(boxes, timestamp=timestamp)
        link.send(boxes, state.current_state(), plan, CURRENT_GOAL.pos)
    turn_ready.clear()
    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img))
    state.update_camera(boxes, timestamp=timestamp, skip_fix=True)
    success = True
    if KNOWN_BOXES and do_update:
        print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        success = state.transform_known_boxes(KNOWN_BOXES)
        print(f"Post-transform state {success=}", state.current_state())

    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img))
    state.update_camera(boxes, timestamp=timestamp, skip_fix=True)

    link.send(boxes, state.current_state(), plan, CURRENT_GOAL.pos)
    return success
    # input("Press enter to continue")
    # print("Scan complete, state", state.current_state())


def state_thread(
    state: KalmanStateFixed,
):
    try:
        with Link(server_ip, server_port) as link:
            cam = aruco_utils.get_camera_picamera()
            need_rescan = False
            for _ in range(N_START_SCANS):
                need_rescan = not circular_scan(cam, state, link, do_update=True)
                scan_ready.set()
            while not stop_program.is_set():
                # print(f"State thread: {state.current_state()}")
                if turn_barrier.n_waiting:
                    try:
                        turn_barrier.wait()  # Synchronise with other thread
                        turn_barrier.wait()  # Turn is complete
                    except:
                        print("Was waiting at turn barrier!")
                        raise

                if re_scan_barrier.n_waiting or need_rescan:
                    while True:
                        try:
                            re_scan_barrier.wait(timeout=0.5)
                            break
                        except threading.BrokenBarrierError:
                            re_scan_barrier.reset()
                            if turn_barrier.n_waiting:
                                break
                        except:
                            print("Was waiting at re_scan_barrier!")
                            raise

                    if turn_barrier.n_waiting:
                        continue
                    need_rescan = not circular_scan(cam, state, link, do_update=True)
                    scan_ready.set()
                    if need_rescan:
                        print("WARNING: re-scan failed, box positions seem to be wrong")

                img = cam.capture_array()
                timestamp = time.time()

                markers = sample_markers(img)
                if not markers:
                    link.send([], state.current_state(), plan, CURRENT_GOAL.pos)
                    continue
                boxes = dedup_camera(markers)
                # print(f"Found boxes! {timestamp}")
                state.update_camera(boxes, timestamp=timestamp)

                est_state = state.current_state()
                link.send(boxes, est_state, plan, CURRENT_GOAL.pos)
    finally:
        print("State exiting!")
        scan_ready.set()
        stop_program.set()


def path_plan(robot, state, original_goal):
    global plan, CURRENT_GOAL
    CURRENT_GOAL = original_goal
    last_scan_time = time.time()
    goal_update_time = last_scan_time
    old_expected_idx = None
    need_rescan = False
    while not stop_program.is_set():
        if (
            (time.time() - last_scan_time > AUTO_SCAN_INTERVAL)
            or re_scan_barrier.n_waiting
            or need_rescan
        ):
            re_scan_barrier.wait()
            robot.spin_left(state=state, event=turn_ready)
            turn_ready.set()
            scan_ready.wait()
            scan_ready.clear()
            updated_goal = state.project_goal(original_goal.pos)
            if updated_goal is None:
                need_rescan = True
                continue
            CURRENT_GOAL = Node(updated_goal)
            last_scan_time = goal_update_time = time.time()

        if time.time() - goal_update_time > GOAL_UPDATE_INTERVAL:
            updated_goal = state.project_goal(original_goal.pos)
            if updated_goal is None:
                need_rescan = True
                continue
            CURRENT_GOAL = Node(updated_goal)
            goal_update_time = time.time()

        state.set_move_predictor(Stopped())
        est_state = state.current_state()
        print(
            f"We think the angle is {est_state.angle}"
            f" {math.degrees(est_state.angle)}"
        )
        if np.linalg.norm(np.asarray(est_state.robot) - CURRENT_GOAL.pos) < 20:
            print("At goal!")
            break

        plan = RRT.generate_plan(
            landmarks=est_state.boxes,
            start=est_state.robot,
            goal=CURRENT_GOAL,
            max_iter=2_000,
            clip_first=300 if old_expected_idx is None else 800,
            old_plan=plan,
            old_expected_idx=old_expected_idx,
            changed_radia={5: 10},  # tODO: remove
        )
        if plan is None:
            print("Couldn't find a plan!")
            need_rescan = True
            continue
        print("plan:", plan)
        # input("postplan.")

        angle, _dist = state.propose_movement(plan[-2])
        turn_barrier.wait()
        # input(f"Ready for turn segment... ({math.degrees(angle)})")
        robot.turn(angle, state=state)
        turn_barrier.wait()
        if (abs(angle) % 360) < math.radians(70):
            # input("Ready for go segment...")
            robot.go_forward(plan[-1], plan[-2], state=state)
            old_expected_idx = 1
        else:
            old_expected_idx = 0
    plan = None


def main_thread():
    global plan
    state = KalmanStateFixed(n_boxes=9)
    with CalibratedRobot() as robot:
        t = threading.Thread(target=state_thread, args=[state])
        t.start()
        for _ in range(N_START_SCANS):
            robot.spin_left(state=state, event=turn_ready)
            turn_ready.set()
            scan_ready.wait()
            scan_ready.clear()

        assert state.current_state().boxes, "No boxes found in scan! Can't plan!"
        time.sleep(0.5)
        path_plan(robot, state, Node(np.array([0, -400])))
        path_plan(robot, state, Node(np.array([280, 940])))

    stop_program.set()
    t.join()


if __name__ == "__main__":
    try:
        main_thread()
    finally:
        stop_program.set()
        print("Exiting...")

# TODO: check boxes out of position and then rescan
