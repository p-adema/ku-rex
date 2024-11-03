import math
import threading
import time

import numpy as np

import global_state
import robot
from kalman_state_fixed import KalmanStateFixed
from movement_predictors import LinearInterpolation, LinearTurn, Stopped

SLOW_CIRCLE_DUR = 5.5

ROBOT_SPEED = 430  # millimeters per second 450
ROBOT_ROTATION = 0.0078  # seconds per degree 0.0078
ROBOT_CAL_20 = 17.3
left_speed = 66
right_speed = 63

arr_times = np.array(
    [0.0, 0.02, 0.04, 0.05, 0.06, 0.08, 0.1, 0.12, 0.14, 0.15, 0.16, 0.18, 0.2]
)
arr_left_angles = np.array(
    [0.0, 0.4, 1.6, 2.2, 3.4, 6.0, 10.4, 16.0, 21.2, 22.0, 24.4, 27.8, 31.8]
)
arr_right_angles = np.array(
    [0.0, 0.4, 1.6, 2.8, 3.6, 6.2, 10.4, 15.8, 20.8, 23.2, 25.2, 27.4, 30.4]
)


DEBUG = True


class CalibratedRobot:
    def __init__(self):
        self.arlo: robot.Robot | None = None

    def __enter__(self):
        self.arlo = robot.Robot()
        return self

    def go_forward(
        self, start_pos: np.ndarray, end_pos: np.ndarray, state: KalmanStateFixed = None
    ) -> None:
        dist = np.linalg.norm(end_pos - start_pos)
        sleep_dur = dist / ROBOT_SPEED
        start = self.arlo.go(+left_speed, +right_speed)

        if DEBUG:
            print(f"Moving {dist}mm, sleeping {sleep_dur}")

        if state is not None:
            state.set_move_predictor(
                LinearInterpolation(start_pos, end_pos, start, sleep_dur)
            )

        pre_deadline = start + sleep_dur - 0.2
        looking_left = True
        front_dist = side_dist = self.arlo.read_front_ping_sensor()
        while (time.time() < pre_deadline) and (front_dist > 300) and (side_dist > 100):
            looking_left = not looking_left
            front_dist = self.arlo.read_front_ping_sensor()
            if looking_left:
                side_dist = self.arlo.read_left_ping_sensor()
            else:
                side_dist = self.arlo.read_right_ping_sensor()
            time.sleep(0.1)

        if (front_dist > 300) and (side_dist > 100):
            remaining = (start + sleep_dur) - time.time()
            if remaining > 0:
                time.sleep(remaining)

            self.arlo.stop()
            return

        stop_time = self.arlo.stop()
        state.set_move_predictor(Stopped(), cancelled_at=stop_time)

        if side_dist <= 100:
            did_turn = False
            try:
                global_state.turn_barrier.wait(timeout=1)
                self.dodge_side(looking_left)
                did_turn = True
                global_state.turn_barrier.wait(timeout=1)
            except threading.BrokenBarrierError:
                # If we somehow call this from the state_thread
                if not did_turn:
                    self.dodge_side(looking_left)

    def spin_left(
        self,
        *,
        state: KalmanStateFixed,
        event: threading.Event,
        cancel: threading.Event = None,
    ):
        self.arlo.go(+42, -40, t=0.3)
        start = self.arlo.go(-42, +40)
        time.sleep(0.405)
        state.set_move_predictor(LinearTurn(2 * np.pi, start, SLOW_CIRCLE_DUR))
        event.set()
        if cancel is None:
            time.sleep((start + SLOW_CIRCLE_DUR) - time.time())
        else:
            cancel.wait(timeout=(start + SLOW_CIRCLE_DUR) - time.time())
        stop_time = self.arlo.stop()
        state.set_move_predictor(Stopped(), cancelled_at=stop_time)

    def turn_left(
        self,
        theta_deg: float,
        state: KalmanStateFixed = None,
        stop: bool = True,
        actual_start: float = None,
    ) -> None:
        if theta_deg < 30:
            sleep_dur = np.interp(theta_deg, arr_left_angles, arr_times)
        else:
            sleep_dur = 0.008031241606277339 * theta_deg - 0.0453796789728319

        if theta_deg == 360:
            sleep_dur *= 1.03  # 1.015

        # if sleep_dur < 0.05:
        #     return

        start = self.arlo.go(-66, +64)
        if actual_start is not None:
            start = actual_start

        if DEBUG:
            print(f"Turning {theta_deg} deg left, sleeping {sleep_dur}")

        if state is not None:
            state.set_move_predictor(
                LinearTurn(math.radians(theta_deg), start, sleep_dur)
            )

        remaining = (start + sleep_dur) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        if stop:
            self.arlo.stop()
        # final = time.time() - start - sleep_dur
        # print(f"Turn error {final:.4f}")

    def turn_right(self, theta_deg: float, state: KalmanStateFixed = None):
        if theta_deg < 30:
            sleep_dur = np.interp(theta_deg, arr_right_angles, arr_times)
        else:
            sleep_dur = 0.007970288435463647 * theta_deg - 0.044264287596813466

        # if sleep_dur < 0.05:
        #     return

        start = self.arlo.go(+66, -64)

        if DEBUG:
            print(f"Turning {theta_deg} deg right, sleeping {sleep_dur}")

        if state is not None:
            state.set_move_predictor(
                LinearTurn(-math.radians(theta_deg), start, sleep_dur)
            )

        remaining = (start + sleep_dur) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.arlo.stop()
        # final = time.time() - start - sleep_dur
        # print(f"Turn error {final:.4f}")

    def turn(self, theta_rad: float, state: KalmanStateFixed = None):
        # Make sure we don't turn more than 180 degrees
        theta_deg = math.degrees(theta_rad) % 360
        # print(f"Turn {theta_rad=} {theta_deg=} (left)")
        if theta_deg < 180:
            return self.turn_left(theta_deg, state=state)
        else:
            return self.turn_right(360 - theta_deg, state=state)

    def stop(self):
        print("Stopping robot")
        self.arlo.stop()

    def median_ping_left(self, n=3) -> int:
        return sorted(self.arlo.read_left_ping_sensor() for _ in range(n))[n // 2 + 1]

    def median_ping_front(self, n=3) -> int:
        return sorted(self.arlo.read_front_ping_sensor() for _ in range(n))[n // 2 + 1]

    def median_ping_right(self, n=3) -> int:
        return sorted(self.arlo.read_right_ping_sensor() for _ in range(n))[n // 2 + 1]

    def fast_forward(self, t: float = 0.3):
        front_dist = side_dist = self.arlo.read_front_ping_sensor()
        deadline = time.time() + t
        looking_left = True
        self.arlo.go(+106, +103)
        while (time.time() < deadline) and (front_dist > 500) and (side_dist > 200):
            looking_left = not looking_left
            front_dist = self.arlo.read_front_ping_sensor()
            if looking_left:
                side_dist = self.arlo.read_left_ping_sensor()
            else:
                side_dist = self.arlo.read_right_ping_sensor()

        if side_dist <= 200 or front_dist <= 500:
            self.arlo.stop()
            self.dodge_side(looking_left)

    def seek_forward(self, cam_dist: float) -> float:
        side_dist = front_dist = self.arlo.read_front_ping_sensor()
        deadline = time.time() + cam_dist / ROBOT_SPEED
        self.arlo.go(+left_speed, +right_speed)
        print(f"seek_forward: {cam_dist=}")
        looking_left = True
        while (time.time() < deadline) and (front_dist > 300) and (side_dist > 100):
            looking_left = not looking_left
            front_dist = self.arlo.read_front_ping_sensor()
            if looking_left:
                side_dist = self.arlo.read_left_ping_sensor()
            else:
                side_dist = self.arlo.read_right_ping_sensor()

        stop_time = self.arlo.stop()
        stopped_early = stop_time - deadline < -0.3
        print(f"seek_forward: {stopped_early=} {front_dist=} {side_dist}")

        if not stopped_early or side_dist > 100:
            return cam_dist

        self.dodge_side(looking_left)
        return 0

    def dodge_side(self, looking_left):
        angle = math.radians(90)
        if looking_left:
            angle *= -1
        self.turn(angle)
        time.sleep(0.2)
        ping = self.arlo.read_front_ping_sensor()
        if ping > 1_000:
            self.arlo.go(+left_speed, +right_speed, t=1.4)
            self.arlo.stop()
            time.sleep(0.2)
        elif ping > 500:
            self.arlo.go(+left_speed, +right_speed, t=0.7)
            self.arlo.stop()
            time.sleep(0.2)
        self.turn(-angle)

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.arlo.stop()
        self.arlo = None
