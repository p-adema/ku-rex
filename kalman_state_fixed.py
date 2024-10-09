import math
import queue
import threading
import time

import numpy as np

from box_types import Box, StateEstimate
from movement_predictors import MovementPredictor, Stopped


# State format: (2 + n_boxes * 2)
# RobotX, RobotY, Box1X, Box1Y, ...
# Measurement format: (n_boxes * 2)
# Box1X, Box1Y, ...
class KalmanStateFixed:
    def __init__(self, n_boxes: int = 4):
        self._n_boxes = n_boxes
        self._transition_mat = np.identity(2 + n_boxes * 2, dtype=float)
        self._transition_covar = np.diag([5.0, 5.0] + [0.1] * (n_boxes * 2))

        # We are certain that we start at (0, 0), facing forwards
        # We are very uncertain about the location of the boxes
        self._cur_mean = np.zeros((2 + n_boxes * 2,), dtype=float)
        self._cur_covar = np.diag([0.0, 0.0] + [1e9] * (n_boxes * 2))

        # A matrix of (Measurement) x (State)
        # gets left-multiplied with the state to produce the predicted measurement
        self._measurement_mat = np.zeros((n_boxes * 2, 2 + n_boxes * 2))
        # (Measurement) x (Measurement) covariance matrix.
        # We're fairly certain about the camera measurements
        self._measurement_covar = np.diag(np.repeat(20, n_boxes * 2))

        self._buf_identity = np.identity(2 + self._n_boxes * 2)
        self._buf_measurement = np.zeros((self._n_boxes * 2,))
        self._timestamp = time.time()
        self._lock = threading.Lock()
        self._movements = queue.Queue()
        self._angle = math.radians(90.0)
        self._move: MovementPredictor = Stopped()

    def transform_known_boxes(self, boxes: list[Box]):
        assert len(boxes) > 0, "Must have at least one box"
        root_box, *turn_boxes = boxes
        assert root_box.x == 0 and root_box.y == 0, "Root box (first) should be origin"
        variance = np.diag(self._cur_covar).reshape((-1, 2)).mean(1)
        assert variance[root_box.id] < 200, "Root box must be spotted"
        translation = -self._cur_mean[root_box.id * 2 : root_box.id * 2 + 2].reshape(
            (1, 2)
        )
        positions = self._cur_mean.reshape((-1, 2)) + translation
        if not turn_boxes:
            return

        angle_estimates = []
        missed_boxes = []
        for box in turn_boxes:
            if variance[box.id] > 200:
                missed_boxes.append(box)
                continue
            box_current = positions[box.id]
            angle_estimates.append(
                (np.arctan2(box.y, box.x) - np.arctan2(box_current[1], box_current[0]))
                % (math.pi * 2)
            )
        assert angle_estimates, "None of the angle boxes spotted!"

        angle = np.mean(angle_estimates)

        print(f"Angle is ~ {math.degrees(angle):.0f} degrees")
        rot_mat = self.rotation_matrix(angle)
        self._angle = (self._angle + angle) % (math.pi * 2)
        corrected_positions = (positions @ rot_mat).flatten()
        for box in missed_boxes:
            corrected_positions[box.id] = box.x, box.y
            self._cur_covar[box.x * 2, box.x * 2] = 20
            self._cur_covar[box.x * 2 + 1, box.x * 2 + 1] = 20

        self._cur_mean = corrected_positions

    def update_camera(self, boxes: list[Box], timestamp: float):
        assert isinstance(boxes, list), f"Type! {boxes}"
        if not boxes:
            return

        assert isinstance(boxes[0], Box), f"Type inner! {boxes[0]}"
        assert len(set(b.id for b in boxes)) == len(boxes), f"Duplicate: {boxes}"
        assert all(0 < b.id <= self._n_boxes for b in boxes), f"Strange ID: {boxes}"
        assert max(max(abs(b.x), abs(b.y)) for b in boxes) < 10_000, f"OOB: {boxes}"

        with self._lock:
            self._buf_measurement.fill(0)
            self._measurement_mat.fill(0)

            for box in boxes:
                self._buf_measurement[(box.id - 1) * 2 : box.id * 2] = box
                self._buf_measurement[box.id * 2 - 1] -= 225
                # Only valid measurements should be taken into account
                self._measurement_mat[-2 + box.id * 2, box.id * 2] = 1
                self._measurement_mat[-1 + box.id * 2, 1 + box.id * 2] = 1
                self._measurement_mat[-2 + box.id * 2, 0] = -1
                self._measurement_mat[-1 + box.id * 2, 1] = -1

            angle = self._angle - math.radians(90)

            rot_mat = self.rotation_matrix(angle)
            measurements = self._buf_measurement.reshape((self._n_boxes, 2)) @ rot_mat

            duration = timestamp - self._timestamp
            assert (
                0 < duration < 50
            ), f"Long duration! {timestamp}-{self._timestamp}={duration:.1f}"

            pred_move, pred_turn = self._move.predict(self._timestamp, timestamp)
            self._angle += pred_turn
            self._cur_mean[:2] += pred_move

            self._timestamp = timestamp

            self._update_camera(measurements.flatten(), duration)

    @staticmethod
    def rotation_matrix(angle):
        return np.array(
            [
                [math.cos(angle), math.sin(angle)],
                [math.sin(-angle), math.cos(angle)],
            ]
        )

    def _update_camera(self, measurement: np.ndarray, duration: float):
        assert measurement.shape == (self._n_boxes * 2,), "Wrong measurement shape!"
        assert 0 < duration < 5, f"Very long duration! ({duration:.2f})"

        partial_trans_covar = self._transition_covar * duration

        pred_mean = self._transition_mat @ self._cur_mean
        pred_covar = (
            self._transition_mat @ self._cur_covar @ self._transition_mat.T
            + partial_trans_covar
        )
        pred_covar_half_update = pred_covar @ self._measurement_mat.T
        gain = pred_covar_half_update @ np.linalg.inv(
            self._measurement_mat @ pred_covar_half_update + self._measurement_covar
        )

        self._cur_mean = pred_mean + gain @ (
            measurement - self._measurement_mat @ pred_mean
        )
        self._cur_covar = (
            self._buf_identity - gain @ self._measurement_mat
        ) @ pred_covar

    def set_move_predictor(self, move: MovementPredictor):
        with self._lock:
            pred_move, pred_turn = self._move.predict(self._timestamp, float("inf"))
            self._angle += pred_turn
            self._cur_mean[:2] += pred_move
            self._move = move

    def current_state(self, permissible_variance: float = 200.0) -> StateEstimate:
        with self._lock:
            boxes = []
            var_diag = np.diag(self._cur_covar)
            for box in range(1, self._n_boxes + 1):
                if var_diag[box * 2 : 1 + box * 2].max() < permissible_variance:
                    boxes.append(
                        Box(
                            id=box,
                            x=self._cur_mean[box * 2],
                            y=self._cur_mean[box * 2 + 1],
                        )
                    )

            exp_move, exp_turn = self._move.predict(self._timestamp, time.time())
            pos = self._cur_mean[:2] + exp_move
            return StateEstimate(Box(0, *pos), boxes, self._angle + exp_turn)

    def propose_movement(self, target: np.ndarray, pos: np.ndarray = None):
        with self._lock:
            if pos is None:
                pos = self._cur_mean[:2].flatten()
            diff = target.flatten() - pos
            target_angle = np.arctan2(diff[1], diff[0])
            turn = target_angle - self._angle
            print(f"Turning {turn}")

            dist = np.linalg.norm(diff)
            return turn, dist

    def set_pos(self, pos: np.ndarray = None, turn: float = None):
        with self._lock:
            if pos is not None:
                print("Hard setting position to ", pos)
                print("Pre-set", self._cur_mean)
                self._cur_mean[:2] = pos.flatten()
                print("Post-set", self._cur_mean)
                self._cur_covar[0, 0] = 20
                self._cur_covar[1, 1] = 20
            if turn is not None:
                self._angle = (self._angle + turn) % (math.pi * 2)


def main_test():
    rng = np.random.default_rng(seed=1)
    coords = np.array([[100.0, 100], [200, 200]])

    num_measurements = 11
    validities = [(True, True)] * num_measurements
    speeds = np.cumsum(np.repeat(5, num_measurements))
    noise = 10

    state = KalmanStateFixed(2)
    start = time.perf_counter()

    # state.debug_print()
    for vs, sp in zip(validities, speeds):
        boxes = []
        for box, v in enumerate(vs):
            if v:
                pos = coords[box] + rng.normal(scale=noise, size=(2,))
                pos[1] += sp
                boxes.append(Box(box + 1, *pos))

        # print("Measurement:", boxes)
        state.update_camera(boxes, force_duration=0.2)
        # state.debug_print()
        print("State:      ", state.current_state())

    print("True y:     ", coords[:, 1] + speeds[-1])
    print(f"took {time.perf_counter() - start} seconds")


if __name__ == "__main__":
    main_test()
