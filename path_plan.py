from __future__ import annotations

import random
import threading
import time

import numpy as np

import constants
import global_state
from box_types import Box
from kalman_state_fixed import KalmanStateFixed
from move_calibrated import CalibratedRobot
from movement_predictors import Stopped
from rrt_landmarks import RRT, Node
from state_thread import state_thread


def path_plan(
    robot: CalibratedRobot,
    state: KalmanStateFixed,
    original_goal: Node,
    changed_radia: dict[int, float] = None,
) -> bool:
    if changed_radia is None:
        changed_radia = {}
    global_state.CURRENT_GOAL = original_goal
    global_state.CURRENT_PLAN = None
    last_scan_time = time.time()
    old_expected_idx = None
    need_rescan = True
    plan_iters = 2_000
    print("Asking for rescan: initial rescan")
    global_state.target_line_of_sight.clear()
    global_state.scan_failed_go_back.clear()
    while not global_state.stop_program.is_set():
        if global_state.target_line_of_sight.is_set():
            return False
        if time.time() - last_scan_time > constants.AUTO_SCAN_INTERVAL:
            print("Asking for rescan: duration")
            need_rescan = True

        if global_state.scan_failed_go_back.is_set():
            global_state.turn_barrier.wait(timeout=3)
            robot.dodge_side(random.random() > 0.3)
            global_state.turn_barrier.wait(timeout=3)
            global_state.scan_failed_go_back.clear()

        if (
            global_state.re_scan_barrier.n_waiting
            or need_rescan
            or (state.known_badness() > 400)
        ):
            need_rescan = False
            synchronised_rescan_main(robot, state)
            last_scan_time = time.time()
            continue

        state.set_move_predictor(Stopped())
        est_state = state.current_state()
        if not est_state.boxes:
            need_rescan = True
            continue

        goal_dist = np.linalg.norm(
            np.asarray(est_state.robot) - global_state.CURRENT_GOAL.pos
        )

        if global_state.target_line_of_sight.is_set():
            return False
        if goal_dist < 500:
            print("path_plan thinks we're at the goal")
            global_state.turn_barrier.wait(timeout=5)  # Enters turn barrier
            robot.turn_left(180, state=state)
            global_state.turn_barrier.wait(timeout=5)  # Exit turn barrier
            robot.go_forward(np.array([0, 0]), np.array([0, 500]), state=state)
            return True

        #  or robot.arlo.read_front_ping_sensor() < 100

        if global_state.target_line_of_sight.is_set():
            return False

        global_state.CURRENT_PLAN = RRT.generate_plan(
            landmarks=est_state.boxes,
            start=est_state.robot,
            goal=global_state.CURRENT_GOAL,
            max_iter=plan_iters,
            clip_first=1_500,  # 300 if old_expected_idx is None else 800,
            old_plan=global_state.CURRENT_PLAN,
            old_expected_idx=old_expected_idx,
            changed_radia=changed_radia,
        )
        if global_state.CURRENT_PLAN is None:
            if plan_iters == 2_000:
                print("Asking for rescan: couldn't find a plan")
                need_rescan = True
                plan_iters = 5_000
            else:
                print("Path plan failed, dodge")
                global_state.turn_barrier.wait(timeout=5)
                robot.turn_left(10 + random.random() * 30, state=state)
                global_state.turn_barrier.wait(timeout=5)
                global_state.scan_failed_go_back.set()
                plan_iters = 2_000
            continue
        else:
            plan_iters = 2_000

        angle, _dist = state.propose_movement(global_state.CURRENT_PLAN[-2])
        if global_state.target_line_of_sight.is_set():
            return False
        global_state.turn_barrier.wait(timeout=5)
        robot.turn(angle, state=state)
        global_state.turn_barrier.wait(timeout=5)
        if global_state.target_line_of_sight.is_set():
            return False
        robot.go_forward(
            global_state.CURRENT_PLAN[-1], global_state.CURRENT_PLAN[-2], state=state
        )
        old_expected_idx = 1
        # if np.linalg.norm(global_state.CURRENT_PLAN[-2] - original_goal.pos) < 500:
        #     print("path_plan thinks we just reached the goal")
        #     return True


def synchronised_rescan_main(robot, state):
    global_state.turn_ready.clear()
    global_state.cancel_spin.clear()
    global_state.re_scan_barrier.wait(timeout=15)
    robot.spin_left(
        state=state, event=global_state.turn_ready, cancel=global_state.cancel_spin
    )
    global_state.turn_ready.set()
    global_state.scan_ready.wait(timeout=5)
    global_state.scan_ready.clear()


def sonar_approach(robot: CalibratedRobot, state: KalmanStateFixed, goal: Box):
    global_state.TARGET_BOX_ID = goal.id
    global_state.cancel_spin.clear()
    global_state.sonar_aligned.clear()
    print("(main) entering sonar_prep 1")
    global_state.sonar_prep_barrier.wait(
        timeout=5
    )  # Allow other thread to capture images
    print("(main) exiting sonar_prep 1")
    global_state.SONAR_ROBOT_HACK = robot
    if global_state.sonar_need_spin.is_set():
        robot.spin_left(
            state=state, event=global_state.turn_ready, cancel=global_state.cancel_spin
        )
        global_state.cancel_spin.set()
    print("(main) entering sonar_prep 2")
    global_state.sonar_prep_barrier.wait(
        timeout=1
    )  # Expect that other thread should be waiting for us
    print("(main) exiting sonar_prep 2")
    print("(main) entering sonar_prep 3")
    global_state.sonar_prep_barrier.wait(
        timeout=30
    )  # Other thread is done with aligning
    print("(main) exiting sonar_prep 3")

    global_state.turn_ready.clear()
    global_state.cancel_spin.clear()
    global_state.scan_ready.clear()
    global_state.SONAR_ROBOT_HACK = None

    return global_state.sonar_aligned.is_set()


def main_thread():
    state = KalmanStateFixed(n_boxes=9)
    with CalibratedRobot() as robot:
        t = threading.Thread(target=state_thread, args=[state])
        t.start()
        avoid_boxes = {box.id: 1_000 for box in constants.KNOWN_BOXES}
        initial_accept = True
        for box_id in (1, 2, 3, 4, 1):
            done = False
            box = constants.KNOWN_BOXES[box_id - 1]
            global_state.TARGET_BOX_ID = box_id
            print(f"\n=== === Visiting box {box_id} === ===\n")
            trust = True
            while not done:
                if (box_id % 2 or not trust) and not initial_accept:
                    _fake_done = path_plan(
                        robot=robot,
                        state=state,
                        original_goal=Node(np.asarray(box)),
                        changed_radia=avoid_boxes | {box.id: -10.0},
                    )
                initial_accept = False
                done = trust = sonar_approach(robot, state, box)
                print(f"MAIN THREAD: {box_id=} {done=}")

    global_state.stop_program.set()
    t.join()
