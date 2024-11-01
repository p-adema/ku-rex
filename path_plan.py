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

AUTO_SCAN_INTERVAL = 1
GOAL_UPDATE_INTERVAL = 1

CURRENT_GOAL = Node(np.array([0, 0]))
CURRENT_PLAN = None
TARGET_BOX_ID = 1
ALLOW_SPIN_INTERRUPTS = False


KNOWN_BOXES = [
    Box(id=1, x=0, y=0),
    Box(id=2, x=0, y=2_650),
    Box(id=3, x=4_000, y=0),
    Box(id=4, x=4_000, y=2_650),
]
N_START_SCANS = 1
SKIP_BOX_MEASUREMENTS = set(range(1, 10))
SONAR_ROBOT_HACK: CalibratedRobot | None = None

stop_program = threading.Event()
turn_ready = threading.Event()
scan_ready = threading.Event()
cancel_spin = threading.Event()
target_line_of_sight = threading.Event()
sonar_aligned = threading.Event()

turn_barrier = threading.Barrier(2)
re_scan_barrier = threading.Barrier(2)
sonar_prep_barrier = threading.Barrier(2)


def circular_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    link: Link,
    do_update: bool = True,
    ignore_far: bool = True,
) -> bool:
    print(f"{ignore_far=}")
    state.force_box_uncertainty(std_dev=250)
    turn_ready.wait(timeout=5)
    turn_ready.clear()
    while not turn_ready.is_set():
        img = cam.capture_array()
        timestamp = time.time()
        markers = sample_markers(img)
        if any(marker.id == TARGET_BOX_ID for marker in markers):
            if ALLOW_SPIN_INTERRUPTS:
                cancel_spin.set()
                input("Interrupting spin")
            target_line_of_sight.set()

        boxes = dedup_camera(markers, skip=SKIP_BOX_MEASUREMENTS)
        state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)
        link.send(boxes, state.current_state(), CURRENT_PLAN, CURRENT_GOAL.pos)
    turn_ready.clear()
    img = cam.capture_array()
    timestamp = time.time()
    markers = sample_markers(img)
    if any(marker.id == TARGET_BOX_ID for marker in markers):
        target_line_of_sight.set()
    boxes = dedup_camera(markers, skip=SKIP_BOX_MEASUREMENTS)
    state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)
    success = True
    if KNOWN_BOXES and do_update and not cancel_spin.is_set():
        # print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        success = state.transform_known_boxes(KNOWN_BOXES, center_around=TARGET_BOX_ID)
        # print(f"Post-transform state {success=}", state.current_state())

    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img), skip=SKIP_BOX_MEASUREMENTS)
    state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)

    link.send(boxes, state.current_state(), CURRENT_PLAN, CURRENT_GOAL.pos)
    return success
    # input("Press enter to continue")
    # print("Scan complete, state", state.current_state())


def state_thread(
    state: KalmanStateFixed,
):
    global SKIP_BOX_MEASUREMENTS, SONAR_ROBOT_HACK, ALLOW_SPIN_INTERRUPTS
    ignore_far = False
    try:
        with Link(server_ip, server_port) as link:
            cam = aruco_utils.get_camera_picamera()
            need_rescan = False
            # for _ in range(N_START_SCANS):
            #     need_rescan = not circular_scan(cam, state, link, do_update=True)
            #     scan_ready.set()
            while not stop_program.is_set():
                # print(f"State thread: {state.current_state()}")
                if turn_barrier.n_waiting:
                    barriers_passed = 0
                    while barriers_passed < 2:
                        try:
                            turn_barrier.wait(timeout=0.5)
                            barriers_passed += 1
                        except threading.BrokenBarrierError:
                            turn_barrier.reset()
                            if stop_program.is_set():
                                break
                        finally:
                            print("Was waiting at turn_barrier!")

                if re_scan_barrier.n_waiting or need_rescan:
                    while True:
                        try:
                            re_scan_barrier.wait(timeout=0.5)
                            ALLOW_SPIN_INTERRUPTS = True
                            break
                        except threading.BrokenBarrierError:
                            re_scan_barrier.reset()
                            if turn_barrier.n_waiting or stop_program.is_set():
                                break
                        finally:
                            print("Was waiting at re_scan_barrier!")

                    if turn_barrier.n_waiting or stop_program.is_set():
                        continue
                    SKIP_BOX_MEASUREMENTS = set(range(1, 10))
                    need_rescan = not circular_scan(
                        cam,
                        state,
                        link,
                        do_update=True,
                        ignore_far=False,  # ignore_far or state.known_badness() > 350,
                    )
                    ignore_far = True
                    SKIP_BOX_MEASUREMENTS = set(range(1, 10))
                    scan_ready.set()
                    if need_rescan:
                        print("WARNING: re-scan failed, box positions seem to be wrong")

                if sonar_prep_barrier.n_waiting:
                    assert TARGET_BOX_ID is not None
                    print("(state) entering sonar_prep 1")
                    try:
                        sonar_prep_barrier.wait()
                    finally:
                        print("(state) exiting sonar_prep 1")
                    turn_ready.wait(timeout=5)
                    cancel_spin.clear()
                    while not cancel_spin.is_set():
                        img = cam.capture_array()
                        timestamp = time.time()

                        markers = sample_markers(img)
                        # print(f"Found markers! {timestamp=} {markers}")
                        if not markers:
                            link.send(
                                [],
                                state.current_state(),
                                None,
                                CURRENT_GOAL.pos,
                            )
                            continue
                        elif any(marker.id == TARGET_BOX_ID for marker in markers):
                            print("Found sonar target!")
                            cancel_spin.set()
                        else:
                            print("Found boxes but not target")

                        boxes = dedup_camera(markers, skip=SKIP_BOX_MEASUREMENTS)
                        state.update_camera(boxes, timestamp=timestamp)

                        est_state = state.current_state()
                        link.send(boxes, est_state, CURRENT_PLAN, CURRENT_GOAL.pos)

                    if not cancel_spin.is_set():
                        # We couldn't find it :(
                        continue

                    print("(state) entering sonar_prep 2")
                    try:
                        sonar_prep_barrier.wait(timeout=1)
                    finally:
                        print("(state) exiting sonar_prep 2")
                    assert SONAR_ROBOT_HACK is not None
                    last_turn = math.radians(5)
                    deadline = time.time() + 600
                    sonar_aligned.clear()
                    spotted = False
                    while time.time() < deadline and not sonar_aligned.is_set():
                        img = cam.capture_array()
                        boxes = dedup_camera(sample_markers(img))
                        target: Box | None = next(
                            filter(lambda b: b.id == TARGET_BOX_ID, boxes), None
                        )
                        if target is None:
                            SONAR_ROBOT_HACK.turn(-last_turn, state=state)
                            print(f"can't see target {TARGET_BOX_ID}")
                            continue
                        if abs(target.x) < 1:
                            print(f"{target} is pretty good")
                            sonar_aligned.set()
                            break
                        angle = math.radians(90) - math.atan(target.y / abs(target.x))
                        if angle < math.radians(10) and target.y > 2_000:
                            SONAR_ROBOT_HACK.arlo.go(+106, +103, t=1)
                        if angle < math.radians(3):
                            sonar_aligned.set()
                            break
                        if not spotted:
                            spotted = True
                            time.sleep(0.2)
                            continue
                        # input(f"Going to turn about {math.degrees(angle)} deg")

                        if target.x > 0:
                            SONAR_ROBOT_HACK.turn(angle, state=state)
                        else:
                            SONAR_ROBOT_HACK.turn(angle, state=state)
                        break

                    print("(state) entering sonar_prep 3")
                    try:
                        sonar_prep_barrier.wait(timeout=5)
                    finally:
                        print("(state) exiting sonar_prep 3")

                img = cam.capture_array()
                timestamp = time.time()
                print(end=".")

                markers = sample_markers(img)
                if not markers:
                    link.send([], state.current_state(), CURRENT_PLAN, CURRENT_GOAL.pos)
                    continue
                if any(marker.id == TARGET_BOX_ID for marker in markers):
                    target_line_of_sight.set()
                boxes = dedup_camera(markers, skip=SKIP_BOX_MEASUREMENTS)

                print(f"Found boxes! {timestamp=} {boxes}")
                state.update_camera(boxes, timestamp=timestamp)

                est_state = state.current_state()
                link.send(boxes, est_state, CURRENT_PLAN, CURRENT_GOAL.pos)
    finally:
        print("State exiting!")
        scan_ready.set()
        stop_program.set()


def path_plan(
    robot: CalibratedRobot,
    state: KalmanStateFixed,
    original_goal: Node,
    changed_radia: dict[int, float] = None,
) -> bool:
    global CURRENT_PLAN, CURRENT_GOAL
    if changed_radia is None:
        changed_radia = {}
    CURRENT_GOAL = original_goal
    CURRENT_PLAN = None
    last_scan_time = time.time()
    # goal_update_time = last_scan_time
    old_expected_idx = None
    need_rescan = True
    plan_iters = 2_000
    print("Asking for rescan: initial rescan")
    target_line_of_sight.clear()
    while not stop_program.is_set():
        if target_line_of_sight.is_set():
            return True
        if (
            (time.time() - last_scan_time > AUTO_SCAN_INTERVAL)
            or re_scan_barrier.n_waiting
            or need_rescan
            or (state.known_badness() > 400)
        ):
            need_rescan = False
            turn_ready.clear()
            cancel_spin.clear()
            re_scan_barrier.wait()
            robot.spin_left(state=state, event=turn_ready, cancel=cancel_spin)
            turn_ready.set()
            scan_ready.wait()
            scan_ready.clear()
            last_scan_time = time.time()

        state.set_move_predictor(Stopped())
        est_state = state.current_state()
        goal_dist = np.linalg.norm(np.asarray(est_state.robot) - CURRENT_GOAL.pos)
        if goal_dist < 500 or robot.arlo.read_front_ping_sensor() < 100:
            print("At goal, but not spotted :(")
            turn_barrier.wait(timeout=5)
            robot.turn_left(180, state=state)
            robot.go_forward(np.array([0, 0]), np.array([0, 500]), state=state)
            return False

        if target_line_of_sight.is_set():
            return True

        CURRENT_PLAN = RRT.generate_plan(
            landmarks=est_state.boxes,
            start=est_state.robot,
            goal=CURRENT_GOAL,
            max_iter=plan_iters,
            clip_first=300 if old_expected_idx is None else 800,
            old_plan=CURRENT_PLAN,
            old_expected_idx=old_expected_idx,
            changed_radia=changed_radia,  # tODO: remove
        )
        if CURRENT_PLAN is None:
            print("Asking for rescan: couldn't find a plan!")
            need_rescan = True
            plan_iters = 10_000
            continue
        else:
            plan_iters = 2_000
        # print("plan:", plan)
        # input("postplan.")

        angle, _dist = state.propose_movement(CURRENT_PLAN[-2])
        if target_line_of_sight.is_set():
            return True
        turn_barrier.wait(timeout=5)
        # input(f"Ready for turn segment... ({math.degrees(angle)})")
        robot.turn(angle, state=state)
        turn_barrier.wait(timeout=5)
        if target_line_of_sight.is_set():
            return True
        # if (abs(angle) % 360) < math.radians(70):
        robot.go_forward(CURRENT_PLAN[-1], CURRENT_PLAN[-2], state=state)
        old_expected_idx = 1
        # else:
        #     old_expected_idx = 0


def sonar_approach(robot: CalibratedRobot, state: KalmanStateFixed, goal: Box):
    global TARGET_BOX_ID, SONAR_ROBOT_HACK
    turn_barrier.wait()
    angle, _dist = state.propose_movement(np.asarray(goal))

    robot.turn(angle - np.radians(20), state=state)
    TARGET_BOX_ID = goal.id
    turn_barrier.wait()
    # input("Pre-spin")
    cancel_spin.clear()
    print("(main) entering sonar_prep 1")
    sonar_prep_barrier.wait(timeout=5)  # Allow other thread to capture images
    print("(main) exiting sonar_prep 1")
    robot.spin_left(state=state, event=turn_ready, cancel=cancel_spin)
    SONAR_ROBOT_HACK = robot
    try:
        print("(main) entering sonar_prep 2")
        sonar_prep_barrier.wait(timeout=1)  # Allow other thread to start aligning sonar
    except threading.BrokenBarrierError:
        print("(main) ERROR exiting sonar_prep 2")
        cancel_spin.set()
        sonar_prep_barrier.reset()
        return False
    print("(main) exiting sonar_prep 2")
    # input("Post-spin")
    print("(main) entering sonar_prep 3")
    sonar_prep_barrier.wait(timeout=100)  # Other thread is done with aligning
    print("(main) exiting sonar_prep 3")
    if not sonar_aligned.is_set():
        print("Sonar alignment failed :(")
        return False
    SONAR_ROBOT_HACK = None

    # input(f"Pre-seek {state.current_state()}")
    turn_barrier.wait(timeout=1)
    moved = robot.seek_forward(target_dist=200, max_dist=2_500)
    if moved == 0:
        turn_barrier.wait(timeout=1)
        print("Failed to seek forwards!")
        return False
    # input(f"Post-seek ({moved})")
    robot.turn(np.pi, state=state)
    robot.go_forward(np.array([0, 0]), np.array([0, moved]))
    # input(f"Done {state.current_state()}")
    turn_ready.clear()
    cancel_spin.clear()
    scan_ready.clear()
    turn_barrier.wait(timeout=5)
    TARGET_BOX_ID = None
    return True


def main_thread():
    global CURRENT_PLAN, TARGET_BOX_ID
    state = KalmanStateFixed(n_boxes=9)
    with CalibratedRobot() as robot:
        t = threading.Thread(target=state_thread, args=[state])
        t.start()
        # for _ in range(N_START_SCANS):
        #     robot.spin_left(state=state, event=turn_ready)
        #     turn_ready.set()
        #     scan_ready.wait()
        #     scan_ready.clear()

        # assert state.current_state().boxes, "No boxes found in scan! Can't plan!"
        # time.sleep(0.5)
        avoid_boxes = {box.id: 1_000 for box in KNOWN_BOXES}
        for box_id in (1, 2, 3, 4, 1):
            done = False
            box = KNOWN_BOXES[box_id - 1]
            TARGET_BOX_ID = box_id
            print(f"Visiting {box}")
            trust = True
            while not done:
                if box_id % 2 or not trust:
                    path_plan(
                        robot=robot,
                        state=state,
                        original_goal=Node(np.asarray(box)),
                        changed_radia=avoid_boxes | {box.id: -10.0},
                    )
                done = trust = sonar_approach(robot, state, box)

    stop_program.set()
    t.join()


if __name__ == "__main__":
    try:
        main_thread()
    finally:
        stop_program.set()
        print("Exiting...")

# TODO:
#   [/] stop path plan on line-of-sight
#   [/] path-plan directly to boxes
#   don't do updates after the first
#   [/]sonar better alignment
