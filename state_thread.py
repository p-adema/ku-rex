from __future__ import annotations

import math
import threading
import time

import picamera2

import aruco_utils
import global_state
from aruco_utils import sample_markers
from box_types import Box, dedup_camera
from client_link import Link
from constants import KNOWN_BOXES, server_ip, server_port
from global_state import (
    ALLOW_SPIN_INTERRUPTS,
    CURRENT_GOAL,
    CURRENT_PLAN,
    SKIP_BOX_MEASUREMENTS,
    TARGET_BOX_ID,
    cancel_spin,
    re_scan_barrier,
    scan_ready,
    sonar_aligned,
    sonar_prep_barrier,
    stop_program,
    target_line_of_sight,
    turn_barrier,
    turn_ready,
)
from kalman_state_fixed import KalmanStateFixed


def circular_scan(
    cam: picamera2.Picamera2,
    state: KalmanStateFixed,
    link: Link,
    do_update: bool = True,
    ignore_far: bool = True,
) -> bool:
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
        print("DOING TRANSFORMATION")
        # print("Pre-transform state", state.current_state())
        # input("Press enter to transform")
        success = state.transform_known_boxes(KNOWN_BOXES, center_around=TARGET_BOX_ID)
        # print(f"Post-transform state {success=}", state.current_state())
    else:
        print("SKIPPING TRANSFORMATION")

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
                    state_turn_barrier()

                if re_scan_barrier.n_waiting or need_rescan:
                    scan_ok = syncronised_rescan_state(cam, link, state)
                    if scan_ok is None:
                        continue
                    need_rescan = not scan_ok
                    ignore_far = True
                    scan_ready.set()
                    if need_rescan:
                        print("WARNING: re-scan failed, box positions seem to be wrong")
                    elif not global_state.ALLOW_SPIN_INTERRUPTS:
                        print("Allowing spin interrupts")
                        global_state.ALLOW_SPIN_INTERRUPTS = True

                if sonar_prep_barrier.n_waiting:
                    synchronised_sonar_state(cam, link, state)

                img = cam.capture_array()
                timestamp = time.time()
                print(end=".")

                markers = sample_markers(img)
                if not markers:
                    link.send([], state.current_state(), CURRENT_PLAN, CURRENT_GOAL.pos)
                    continue
                if any(marker.id == TARGET_BOX_ID for marker in markers):
                    target_line_of_sight.set()
                boxes = dedup_camera(markers, skip=global_state.SKIP_BOX_MEASUREMENTS)

                print(f"Found boxes! {timestamp=} {boxes}")
                state.update_camera(boxes, timestamp=timestamp)

                est_state = state.current_state()
                link.send(boxes, est_state, CURRENT_PLAN, CURRENT_GOAL.pos)
    finally:
        print("State exiting!")
        scan_ready.set()
        stop_program.set()


def state_turn_barrier():
    barriers_passed = 0
    while barriers_passed < 2:
        try:
            turn_barrier.wait(timeout=0.5)
            barriers_passed += 1
        except threading.BrokenBarrierError:
            turn_barrier.reset()
            if (
                stop_program.is_set()
                or sonar_prep_barrier.n_waiting
                or re_scan_barrier.n_waiting
            ):
                break
        finally:
            print("Was waiting at turn_barrier!")


def synchronised_sonar_state(cam, link, state):
    assert TARGET_BOX_ID is not None
    print("(state) entering sonar_prep 1")
    try:
        sonar_prep_barrier.wait()
    finally:
        print("(state) exiting sonar_prep 1")
    turn_ready.wait(timeout=5)
    sonar_find_loop(cam, link, state)
    if not target_line_of_sight.is_set():
        # We couldn't find it :(
        print("(state) skipping rest of sonar_prep")
        return

    print("(state) entering sonar_prep 2")
    try:
        sonar_prep_barrier.wait(timeout=1)
    finally:
        print("(state) exiting sonar_prep 2")
    sonar_align_loop(cam, state)

    print("(state) entering sonar_prep 3")
    try:
        sonar_prep_barrier.wait(timeout=5)
    finally:
        print("(state) exiting sonar_prep 3")


def sonar_find_loop(cam, link, state):
    cancel_spin.clear()
    target_line_of_sight.clear()
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
            target_line_of_sight.set()
        else:
            print("Found boxes but not target")

        boxes = dedup_camera(markers, skip=global_state.SKIP_BOX_MEASUREMENTS)
        state.update_camera(boxes, timestamp=timestamp)

        est_state = state.current_state()
        link.send(boxes, est_state, CURRENT_PLAN, CURRENT_GOAL.pos)


def sonar_align_loop(cam, state):
    assert global_state.SONAR_ROBOT_HACK is not None
    last_turn = math.radians(5)
    deadline = time.time() + 600
    sonar_aligned.clear()
    spotted = False
    while time.time() < deadline and not sonar_aligned.is_set():
        img = cam.capture_array()
        boxes = dedup_camera(sample_markers(img))
        target: Box | None = next(filter(lambda b: b.id == TARGET_BOX_ID, boxes), None)
        if target is None:
            global_state.SONAR_ROBOT_HACK.turn(-last_turn, state=state)
            print(f"can't see target {TARGET_BOX_ID}")
            continue
        if abs(target.x) < 1:
            print(f"{target} is pretty good")
            sonar_aligned.set()
            break
        angle = math.radians(90) - math.atan(target.y / abs(target.x))
        if angle < math.radians(10) and target.y > 2_000:
            global_state.SONAR_ROBOT_HACK.arlo.go(+106, +103, t=0.5)
            time.sleep(0.05)
        if angle < math.radians(3):
            sonar_aligned.set()
            break
        if not spotted:
            spotted = True
            time.sleep(0.2)
            continue
        # input(f"Going to turn about {math.degrees(angle)} deg")

        if target.x > 0:
            global_state.SONAR_ROBOT_HACK.turn(angle, state=state)
        else:
            global_state.SONAR_ROBOT_HACK.turn(angle, state=state)
        break


def syncronised_rescan_state(cam, link, state):
    while True:
        try:
            re_scan_barrier.wait(timeout=0.5)
            break
        except threading.BrokenBarrierError:
            re_scan_barrier.reset()
            if turn_barrier.n_waiting or stop_program.is_set():
                break
        finally:
            print("Was waiting at re_scan_barrier!")
    if turn_barrier.n_waiting or stop_program.is_set():
        scan_ok = None
    if scan_ok is not None:
        scan_ok = circular_scan(
            cam,
            state,
            link,
            do_update=True,
            ignore_far=False,  # ignore_far or state.known_badness() > 350,
        )
    return scan_ok
