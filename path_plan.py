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
from movement_predictors import Stopped
from rrt_landmarks import RRT, Node

AUTO_SCAN_INTERVAL = 1
GOAL_UPDATE_INTERVAL = 1

CURRENT_GOAL = Node(np.array([0, 0]))
CURRENT_PLAN = None
CENTOR_AROUND_BOX = 1

KNOWN_BOXES = [
    Box(id=1, x=0, y=0),
    Box(id=2, x=0, y=2_650),
    Box(id=3, x=4_000, y=0),
    Box(id=4, x=4_000, y=2_650),
]
N_START_SCANS = 1
SKIP_BOX_MEASUREMENTS = set(range(1, 10))
SONAR_PREP_TARGET = None

stop_program = threading.Event()
turn_ready = threading.Event()
scan_ready = threading.Event()
cancel_spin = threading.Event()

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
        boxes = dedup_camera(sample_markers(img), skip=SKIP_BOX_MEASUREMENTS)
        state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)
        link.send(boxes, state.current_state(), CURRENT_PLAN, CURRENT_GOAL.pos)
    turn_ready.clear()
    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img), skip=SKIP_BOX_MEASUREMENTS)
    state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)
    success = True
    if KNOWN_BOXES and do_update:
        # print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        success = state.transform_known_boxes(KNOWN_BOXES, center_around=CENTER_AROUND_BOX)
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
    global SKIP_BOX_MEASUREMENTS
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
                        except:
                            print("Was waiting at turn_barrier!")
                            raise

                if re_scan_barrier.n_waiting or need_rescan:
                    while True:
                        try:
                            re_scan_barrier.wait(timeout=0.5)
                            break
                        except threading.BrokenBarrierError:
                            re_scan_barrier.reset()
                            if turn_barrier.n_waiting or stop_program.is_set():
                                break
                        except:
                            print("Was waiting at re_scan_barrier!")
                            raise

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
                    assert SONAR_PREP_TARGET is not None
                    sonar_prep_barrier.wait()
                    turn_ready.wait(timeout=5)
                    cancel_spin.clear()
                    while not cancel_spin.is_set():
                        img = cam.capture_array()
                        timestamp = time.time()

                        markers = sample_markers(img)
                        #print(f"Found markers! {timestamp=} {markers}")
                        if not markers:
                            link.send(
                                [],
                                state.current_state(),
                                None,
                                CURRENT_GOAL.pos,
                            )
                            continue
                        elif any(marker.id == SONAR_PREP_TARGET for marker in markers):
                            print("Found sonar target!")
                            cancel_spin.set()
                        else:
                            print("Found boxes but not target")

                        boxes = dedup_camera(markers, skip=SKIP_BOX_MEASUREMENTS)
                        state.update_camera(boxes, timestamp=timestamp)

                        est_state = state.current_state()
                        link.send(boxes, est_state, CURRENT_PLAN, CURRENT_GOAL.pos)
                    continue

                img = cam.capture_array()
                timestamp = time.time()

                markers = sample_markers(img)
                if not markers:
                    link.send([], state.current_state(), CURRENT_PLAN, CURRENT_GOAL.pos)
                    continue
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
    trust: bool = False,
):
    global CURRENT_PLAN, CURRENT_GOAL
    if changed_radia is None:
        changed_radia = {}
    CURRENT_GOAL = original_goal
    last_scan_time = time.time()
    # goal_update_time = last_scan_time
    old_expected_idx = None
    need_rescan = True
    plan_iters = 2_000
    print("Asking for rescan: initial rescan")
    while not stop_program.is_set():
        if (
            (time.time() - last_scan_time > AUTO_SCAN_INTERVAL)
            or re_scan_barrier.n_waiting
            or need_rescan
            or (state.known_badness() > 400)
        ):
            need_rescan = False
            turn_ready.clear()
            re_scan_barrier.wait()
            robot.spin_left(state=state, event=turn_ready)
            turn_ready.set()
            scan_ready.wait()
            scan_ready.clear()
            # updated_goal = state.project_goal(original_goal.pos)
            # if updated_goal is None:
            #     need_rescan = True
            #     print("Can't update goal, asking for rescan (within rescan)")
            #     continue
            # CURRENT_GOAL = Node(updated_goal)
            #boxes = state.current_state().boxes
            #target_box = next(filter(lambda box: box.id == goal_box_id, boxes), None)
            #if target_box is None:
            #    print("Asking for rescan: Didn't see goal box")
            #    need_rescan = True
            #    continue
            #CURRENT_GOAL = Node(np.asarray(target_box) + goal_box_offset)
            last_scan_time = time.time()

        # if time.time() - goal_update_time > GOAL_UPDATE_INTERVAL:
        #     updated_goal = state.project_goal(original_goal.pos)
        #     if updated_goal is None:
        #         print("Can't update goal, asking for rescan (due to timer)")
        #         need_rescan = True
        #         continue
        #     CURRENT_GOAL = Node(updated_goal)
        #     goal_update_time = time.time()

        state.set_move_predictor(Stopped())
        est_state = state.current_state()
        goal_dist = np.linalg.norm(np.asarray(est_state.robot) - CURRENT_GOAL.pos)
        if goal_dist < 50 or (trust and (600 <= goal_dist <= 1200)):
            print("At goal!")
            break

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
        turn_barrier.wait()
        # input(f"Ready for turn segment... ({math.degrees(angle)})")
        robot.turn(angle, state=state)
        turn_barrier.wait()
        # if (abs(angle) % 360) < math.radians(70):
        robot.go_forward(CURRENT_PLAN[-1], CURRENT_PLAN[-2], state=state)
        old_expected_idx = 1
        # else:
        #     old_expected_idx = 0
    CURRENT_PLAN = None


def sonar_approach(robot: CalibratedRobot, state: KalmanStateFixed, goal: Box):
    global SONAR_PREP_TARGET
    turn_barrier.wait()
    angle, _dist = state.propose_movement(np.asarray(goal))
    #input("Entering sonar approach")

    # Overturn to be to the right of the box
    robot.turn(angle - 20, state=state)
    SONAR_PREP_TARGET = goal.id
    turn_barrier.wait()
    #input("Pre-spin")
    cancel_spin.clear()
    sonar_prep_barrier.wait(timeout=5)
    robot.spin_left(state=state, event=turn_ready, cancel=cancel_spin)
    try:
        turn_barrier.wait(timeout=1)
    except threading.BrokenBarrierError:
        cancel_spin.set()
        turn_barrier.reset()
        return False
    # input("Post-spin")
    angle, _dist = state.propose_movement(np.asarray(goal))
    robot.turn(angle, state=state)
    #input(f"Pre-seek {state.current_state()}")
    moved = robot.seek_forward(target_dist=200, max_dist=2_500)
    if moved == 0:
        print("Failed to seek forwards!")
        turn_barrier.wait(timeout=1)
        return False
    #input(f"Post-seek ({moved})")
    robot.turn(np.pi, state=state)
    robot.go_forward(np.array([0, 0]), np.array([0, moved]))
    #input(f"Done {state.current_state()}")
    turn_ready.clear()
    cancel_spin.clear()
    scan_ready.clear()
    turn_barrier.wait(timeout=5)
    SONAR_PREP_TARGET = None
    return True


def main_thread():
    global CURRENT_PLAN
    global CENTER_AROUND_BOX
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
        in_front = np.array([0, -350])
        avoid_boxes = {box.id: 1_000 for box in KNOWN_BOXES}
        pos_y = np.array([0, 1000])
        neg_y = -pos_y
        corner_3 = np.array([-700, 700])
        corner_1 = np.array([700, 700])
        for num, offset in zip((1, 2, 3, 4, 1), (pos_y, neg_y, pos_y, neg_y, pos_y)):
            done = False
            box = KNOWN_BOXES[num - 1]
            CENTER_AROUND_BOX = num
            print(f"Visiting {box}")
            trust = True
            while not done:
                if num % 2 or not trust:
                    path_plan(
                        robot=robot,
                        state=state,
                        original_goal=Node(np.asarray(box) + offset),
                        changed_radia=avoid_boxes | {box.id: 550.0},
                        trust=trust
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

# TODO: check boxes out of position and then rescan
