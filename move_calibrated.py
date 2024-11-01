import math
import threading
import time

import numpy as np

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

        remaining = (start + sleep_dur) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.arlo.stop()

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

        if sleep_dur < 0.05:
            return

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
        final = time.time() - start - sleep_dur
        print(f"Turn error {final:.4f}")

    def turn_right(self, theta_deg: float, state: KalmanStateFixed = None):
        if theta_deg < 30:
            sleep_dur = np.interp(theta_deg, arr_right_angles, arr_times)
        else:
            sleep_dur = 0.007970288435463647 * theta_deg - 0.044264287596813466

        if sleep_dur < 0.05:
            return

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
        final = time.time() - start - sleep_dur
        print(f"Turn error {final:.4f}")

    def turn(self, theta_rad: float, state: KalmanStateFixed = None):
        # Make sure we don't turn more than 180 degrees
        theta_deg = math.degrees(theta_rad) % 360
        print(f"Turn {theta_rad=} {theta_deg=} (left)")
        if theta_deg < 180:
            return self.turn_left(theta_deg, state=state)
        else:
            return self.turn_right(360 - theta_deg, state=state)

    def stop(self):
        print("Stopping robot")
        self.arlo.stop()

    def seek_forward(self, target_dist: float, max_dist: float) -> int:
        initial_dist = side_dist = front_dist = self.arlo.read_front_ping_sensor()
        if front_dist > max_dist:
            print(f"Seek failed, distance {front_dist} > {max_dist}")
            return 0
        self.arlo.go(+44, +42)
        looking_left = True
        while (front_dist > target_dist + 10) and side_dist > 100:
            front_dist = self.arlo.read_front_ping_sensor()
            if looking_left:
                side_dist = self.arlo.read_left_ping_sensor()
            else:
                side_dist = self.arlo.read_right_ping_sensor()
            looking_left = not looking_left

        self.arlo.stop()
        return initial_dist - self.arlo.read_front_ping_sensor()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.arlo.stop()
        self.arlo = None
