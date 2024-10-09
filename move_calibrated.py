import contextlib
import math
import time

import numpy as np

import robot
from movement_predictors import LinearInterpolation, LinearTurn, MovementPredictor

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


class CalibratedRobot:
    def __init__(self):
        self.arlo = None

    def __enter__(self):
        self.arlo = robot.Robot()
        return self

    @contextlib.contextmanager
    def go_forward(
        self, start_pos: np.ndarray, end_pos: np.ndarray
    ) -> contextlib.AbstractContextManager[MovementPredictor]:
        dist = np.linalg.norm(end_pos - start_pos)
        sleep_dur = dist / ROBOT_SPEED
        start = self.arlo.go(+left_speed, +right_speed)

        yield LinearInterpolation(start_pos, end_pos, start, sleep_dur)

        remaining = (start + sleep_dur) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.arlo.stop()

    @contextlib.contextmanager
    def turn_left(
        self, theta_deg: float
    ) -> contextlib.AbstractContextManager[MovementPredictor]:
        if theta_deg < 30:
            sleep_dur = np.interp(theta_deg, arr_left_angles, arr_times)
        else:
            sleep_dur = 0.008031241606277339 * theta_deg - 0.0453796789728319

        start = self.arlo.go(-66, +64)

        yield LinearTurn(theta_deg, start, sleep_dur)

        remaining = (start + sleep_dur) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.arlo.stop()

    @contextlib.contextmanager
    def turn_right(
        self, theta_deg: float
    ) -> contextlib.AbstractContextManager[MovementPredictor]:
        if theta_deg < 30:
            sleep_dur = np.interp(theta_deg, arr_right_angles, arr_times)
        else:
            sleep_dur = 0.007970288435463647 * theta_deg - 0.044264287596813466

        start = self.arlo.go(+66, -64)

        yield LinearTurn(-theta_deg, start, sleep_dur)

        remaining = (start + sleep_dur) - time.time()
        if remaining > 0:
            time.sleep(remaining)
        self.arlo.stop()

    def turn(
        self, theta_rad: float
    ) -> contextlib.AbstractContextManager[MovementPredictor]:
        # Make sure we don't turn more than 180 degrees
        theta_deg = (math.degrees(theta_rad) + 540) % 360 - 180
        print(f"Turn {theta_rad=} {theta_deg=}")
        if theta_deg > 0:
            return self.turn_left(theta_deg)
        else:
            return self.turn_right(-theta_deg)

    def stop(self):
        print("Stopping robot")
        self.arlo.stop()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.arlo.stop()
        self.arlo = None
