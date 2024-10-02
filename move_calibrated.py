import math
import time

import robot

ROBOT_SPEED = 350  # millimeters per second 450
ROBOT_ROTATION = 0.0078  # seconds per degree 0.0078
ROBOT_CAL_20 = 17.3
left_speed = 66
right_speed = 63
waitTime = 0.041


class CalibratedRobot:
    def __init__(self):
        self.arlo = None

    def __enter__(self):
        self.arlo = robot.Robot()
        return self

    def go_forward(self, dist: float):
        print(f"Move {dist=}")
        sleep_dur = dist / ROBOT_SPEED
        self.arlo.go(+left_speed, +right_speed)
        time.sleep(sleep_dur)
        self.arlo.stop()

    # Turning is imprecise below 20 and above 200
    def turn_right(self, theta_deg: float):
        sleep_dur = ROBOT_ROTATION * theta_deg
        self.arlo.go(+left_speed, -right_speed)
        time.sleep(sleep_dur)
        self.arlo.stop()

    def turn_left(self, theta_deg: float):
        sleep_dur = ROBOT_ROTATION * theta_deg
        self.arlo.go(-left_speed, +right_speed)
        time.sleep(sleep_dur)
        self.arlo.stop()

    def scan_left(self):
        self.turn_left(ROBOT_CAL_20)

    def turn(self, theta_rad: float):
        # Make sure we don't turn more than 180 degrees
        theta_deg = (math.degrees(theta_rad) + 540) % 360 - 180
        print(f"Turn {theta_rad=} {theta_deg=}")
        if abs(theta_deg) < 5:
            return
        if theta_deg > 0:
            self.turn_left(theta_deg)
        else:
            self.turn_right(-theta_deg)

    def stop(self):
        print("Stopping robot")
        self.arlo.stop()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.arlo.stop()
        self.arlo = None
