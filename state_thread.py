from __future__ import annotations

import math
import random
import threading
import time

import numpy as np
import picamera2

import aruco_utils
import global_state
from aruco_utils import sample_markers
from box_types import Box, dedup_camera
from client_link import Link
from constants import KNOWN_BOXES, server_ip, server_port
from kalman_state_fixed import KalmanStateFixed


def circular_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    link: Link,
    do_update: bool = True,
    ignore_far: bool = True,
) -> bool:
    state.force_box_uncertainty(std_dev=250)
    global_state.turn_ready.wait(timeout=5)
    global_state.turn_ready.clear()
    while not global_state.turn_ready.is_set():
        img = cam.capture_array()
        timestamp = time.time()
        markers = sample_markers(img)
        if any(marker.id == global_state.TARGET_BOX_ID for marker in markers):
            if global_state.ALLOW_SPIN_INTERRUPTS:
                global_state.cancel_spin.set()
            global_state.target_line_of_sight.set()

        boxes = dedup_camera(markers, skip=global_state.SKIP_BOX_MEASUREMENTS)
        state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)
        link.send(
            boxes,
            state.current_state(),
            global_state.CURRENT_PLAN,
            global_state.CURRENT_GOAL.pos,
        )
    global_state.turn_ready.clear()
    img = cam.capture_array()
    timestamp = time.time()
    markers = sample_markers(img)
    if any(marker.id == global_state.TARGET_BOX_ID for marker in markers):
        global_state.target_line_of_sight.set()
    boxes = dedup_camera(markers, skip=global_state.SKIP_BOX_MEASUREMENTS)
    state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)
    success = True
    if KNOWN_BOXES and do_update and not global_state.cancel_spin.is_set():
        print("DOING TRANSFORMATION")
        # print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        success = state.transform_known_boxes(
            KNOWN_BOXES, center_around=global_state.TARGET_BOX_ID
        )
        # print(f"Post-transform state {success=}", state.current_state())
    else:
        print("SKIPPING TRANSFORMATION")

    img = cam.capture_array()
    timestamp = time.time()
    boxes = dedup_camera(sample_markers(img), skip=global_state.SKIP_BOX_MEASUREMENTS)
    state.update_camera(boxes, timestamp=timestamp, ignore_far=ignore_far)

    link.send(
        boxes,
        state.current_state(),
        global_state.CURRENT_PLAN,
        global_state.CURRENT_GOAL.pos,
    )
    return success
    # input("Press enter to continue")
    # print("Scan complete, state", state.current_state())


def state_thread(
    state: KalmanStateFixed,
):
    ignore_far = False
    try:
        with Link(server_ip, server_port) as link:
            cam = aruco_utils.get_camera_picamera()
            need_rescan = False
            # for _ in range(N_START_SCANS):
            #     need_rescan = not circular_scan(cam, state, link, do_update=True)
            #     scan_ready.set()
            print("(state) booting up")
            while not global_state.stop_program.is_set():
                # print(f"State thread: {state.current_state()}")
                if global_state.turn_barrier.n_waiting:
                    print("(state) entering turn sync")
                    state_turn_barrier()

                if global_state.re_scan_barrier.n_waiting or need_rescan:
                    print("(state) entering re-scan sync")
                    scan_ok = synchronised_rescan_state(cam, link, state)
                    if scan_ok is None:
                        continue
                    need_rescan = not scan_ok
                    ignore_far = True
                    global_state.scan_ready.set()
                    if need_rescan:
                        print("WARNING: re-scan failed, box positions seem to be wrong")
                    elif not global_state.ALLOW_SPIN_INTERRUPTS:
                        print("Allowing spin interrupts")
                        global_state.ALLOW_SPIN_INTERRUPTS = True

                if global_state.sonar_prep_barrier.n_waiting:
                    print("(state) entering sonar sync")
                    synchronised_sonar_state(cam, link, state)

                img = cam.capture_array()
                timestamp = time.time()
                print(end=".")

                markers = sample_markers(img)
                if not markers:
                    link.send(
                        [],
                        state.current_state(),
                        global_state.CURRENT_PLAN,
                        global_state.CURRENT_GOAL.pos,
                    )
                    continue
                if any(marker.id == global_state.TARGET_BOX_ID for marker in markers):
                    global_state.target_line_of_sight.set()
                boxes = dedup_camera(markers, skip=global_state.SKIP_BOX_MEASUREMENTS)

                print(f"Found boxes! {timestamp=} {boxes}")
                state.update_camera(boxes, timestamp=timestamp)

                est_state = state.current_state()
                link.send(
                    boxes,
                    est_state,
                    global_state.CURRENT_PLAN,
                    global_state.CURRENT_GOAL.pos,
                )
    finally:
        print("State exiting!")
        global_state.scan_ready.set()
        global_state.stop_program.set()


def state_turn_barrier():
    barriers_passed = 0
    while barriers_passed < 2:
        try:
            global_state.turn_barrier.wait(timeout=0.5)
            barriers_passed += 1
        except threading.BrokenBarrierError:
            global_state.turn_barrier.reset()
            if (
                global_state.stop_program.is_set()
                or global_state.sonar_prep_barrier.n_waiting
                or global_state.re_scan_barrier.n_waiting
            ):
                break
        finally:
            print("Was waiting at turn_barrier!")


def synchronised_sonar_state(cam, link, state):
    assert global_state.TARGET_BOX_ID is not None

    global_state.sonar_need_spin.clear()
    if not any(
        marker.id == global_state.TARGET_BOX_ID
        for marker in (sample_markers(cam.capture_array()))
    ):
        global_state.sonar_need_spin.set()
    print("(state) entering sonar_prep 1")
    try:
        global_state.sonar_prep_barrier.wait()
    finally:
        print("(state) exiting sonar_prep 1")

    if global_state.sonar_need_spin.is_set():
        global_state.turn_ready.wait(timeout=5)
        sonar_find_loop(cam, link, state)

    print("(state) entering sonar_prep 2")
    try:
        global_state.sonar_prep_barrier.wait(timeout=1)
    finally:
        print("(state) exiting sonar_prep 2")
    if global_state.target_line_of_sight.is_set():
        cam_dist = sonar_align_loop(cam, state)

        if cam_dist is not None:
            moved = global_state.SONAR_ROBOT_HACK.seek_forward(cam_dist=cam_dist - 400)
            if moved:
                moved = min(moved, 700)  # Don't go more than 70cm back
                global_state.SONAR_ROBOT_HACK.turn(np.pi, state=state)
                global_state.SONAR_ROBOT_HACK.go_forward(
                    np.array([0, 0]), np.array([0, moved])
                )
            else:
                global_state.sonar_aligned.clear()
                global_state.SONAR_ROBOT_HACK.dodge_side(random.random() > 0.3)

        else:
            global_state.sonar_aligned.clear()

    print("(state) entering sonar_prep 3")
    try:
        global_state.sonar_prep_barrier.wait(timeout=5)
    finally:
        print("(state) exiting sonar_prep 3")
    return


def sonar_find_loop(cam, link, state):
    global_state.cancel_spin.clear()
    global_state.target_line_of_sight.clear()
    while (
        not global_state.cancel_spin.is_set() and not global_state.stop_program.is_set()
    ):
        print(end=",")
        img = cam.capture_array()
        timestamp = time.time()

        markers = sample_markers(img)
        # print(f"Found markers! {timestamp=} {markers}")
        if not markers:
            link.send(
                [],
                state.current_state(),
                None,
                global_state.CURRENT_GOAL.pos,
            )
            continue
        elif any(marker.id == global_state.TARGET_BOX_ID for marker in markers):
            print("Found sonar target!")
            global_state.cancel_spin.set()
            time.sleep(0.01)  # Yield thread, stop spin
            global_state.target_line_of_sight.set()

        boxes = dedup_camera(markers, skip=global_state.SKIP_BOX_MEASUREMENTS)
        state.update_camera(boxes, timestamp=timestamp)

        est_state = state.current_state()
        link.send(
            boxes, est_state, global_state.CURRENT_PLAN, global_state.CURRENT_GOAL.pos
        )


def sonar_align_loop(cam, state) -> float | None:
    assert global_state.SONAR_ROBOT_HACK is not None
    last_turn = math.radians(10)
    deadline = time.time() + 2
    global_state.sonar_aligned.clear()
    spotted = False
    approach_fail = 0
    while time.time() < deadline and not global_state.sonar_aligned.is_set():
        img = cam.capture_array()
        boxes = dedup_camera(sample_markers(img))
        target: Box | None = next(
            filter(lambda b: b.id == global_state.TARGET_BOX_ID, boxes), None
        )
        if target is None:
            if approach_fail == 1:
                print("Approach failed, doing 180")
                global_state.SONAR_ROBOT_HACK.turn_left(180)
                global_state.SONAR_ROBOT_HACK.arlo.go(+106, +103, t=0.4)
                global_state.SONAR_ROBOT_HACK.turn_left(180)
                approach_fail += 1
                spotted = False
                continue
            print(
                f"can't see target "
                f"(Box {global_state.TARGET_BOX_ID}) {math.degrees(last_turn)=}"
            )
            global_state.SONAR_ROBOT_HACK.turn(-0.5 * last_turn, state=state)
            spotted = False
            time.sleep(0.1)
            continue
        angle = math.radians(90) - math.atan(target.y / abs(target.x))
        print(f"spotted {target=} ({angle=})")
        if angle < math.radians(7) and target.y > 1_700:
            print("\tapproaching box more closely")
            global_state.SONAR_ROBOT_HACK.fast_forward(t=0.3)
            time.sleep(0.05)
            spotted = False
            if not approach_fail:
                approach_fail = 1
            continue
        if not spotted:
            spotted = True
            time.sleep(0.3)
            continue
        if angle < math.radians(3):
            # sonar_dist = global_state.SONAR_ROBOT_HACK.arlo.read_front_ping_sensor()
            print(
                f"\taccepting sonar calibration, "
                f"angle {np.degrees(angle)}, {target.y=} < 1700"
            )
            global_state.sonar_aligned.set()
            return target.y

        # input(f"Going to turn about {math.degrees(angle)} deg")

        last_turn = angle * 0.5
        if target.x > 0:
            last_turn *= -1
        global_state.SONAR_ROBOT_HACK.turn(last_turn, state=state)
        time.sleep(0.2)
        print("\tturned after spotting, going forward")

    return None


def synchronised_rescan_state(cam, link, state):
    exit_early = False
    while True:
        try:
            global_state.re_scan_barrier.wait(timeout=0.5)
            break
        except threading.BrokenBarrierError:
            global_state.re_scan_barrier.reset()
            if (
                global_state.turn_barrier.n_waiting
                or global_state.stop_program.is_set()
                or global_state.sonar_prep_barrier.n_waiting
            ):
                exit_early = True
                break
        finally:
            print("Was waiting at re_scan_barrier!")
    if exit_early:
        return None

    return circular_scan(
        cam,
        state,
        link,
        do_update=True,
        ignore_far=False,  # ignore_far or state.known_badness() > 350,
    )


# [can't see, doesn't turn]
# [straight for landmark]
#
