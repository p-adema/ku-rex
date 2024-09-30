import math
import queue
import threading
import time

import numpy as np

from box_types import Box, MovementAction, StateEstimate


# State format: (4 + n_boxes * 2)
# RobotX, RobotY, Box1X, Box1Y, ...
# Measurement format: (n_boxes * 2)
# Box1X, Box1Y, ...
class KalmanStateFixed:
    def __init__(self, n_boxes: int = 4):
        self._n_boxes = n_boxes
        self._transition_mat = np.identity(2 + n_boxes * 2, dtype=float)
        self._transition_covar = np.diag([50.0, 50.0] + [0.1] * (n_boxes * 2))

        # We are certain that we start at (0, 0), facing forwards
        # We are very uncertain about the location of the boxes
        self._cur_mean = np.zeros((2 + n_boxes * 2,), dtype=float)
        self._cur_covar = np.diag([0.0, 0.0] + [1e9] * (n_boxes * 2))

        # A matrix of (Measurement) x (State)
        # gets left-multiplied with the state to produce the predicted measurement
        self._measurement_mat = np.hstack(
            (np.tile(np.diag([-1, -1]), n_boxes).T, np.identity(2 * n_boxes))
        )
        # (Measurement) x (Measurement) covariance matrix.
        # We're fairly certain about the camera measurements
        self._measurement_covar = np.diag(np.repeat(20, n_boxes * 2))

        self._buf_identity = np.identity(4 + self._n_boxes * 2)
        self._buf_measurement = np.zeros((self._n_boxes * 2,))
        self._timestamp = time.time()
        self._lock = threading.Lock()
        self._movements = queue.Queue()
        self._angle = 0.0
        print(f"Init state: {self._timestamp=}")

    def update_camera(
        self, boxes: list[Box], timestamp: float = None, force_duration: float = None
    ):
        assert isinstance(boxes, list) and isinstance(boxes[0], Box), f"Type! {boxes}"
        assert len(set(b.id for b in boxes)) == len(boxes), f"Duplicate: {boxes}"
        assert all(0 < b.id <= self._n_boxes for b in boxes), f"Strange ID: {boxes}"
        assert max(max(abs(b.x), abs(b.y)) for b in boxes) < 10_000, f"OOB: {boxes}"
        assert (timestamp is None) + (force_duration is None) == 1, "Exactly one time"

        with self._lock:
            self._buf_measurement.fill(0)

            for box in boxes:
                self._buf_measurement[(box.id - 1) * 2 : box.id * 2] = box
                # Only valid measurements should be taken into account
                self._measurement_mat[box.id * 2, 4 + box.id * 2] = 1
                self._measurement_mat[1 + box.id * 2, 5 + box.id * 2] = 1

            rot_mat = np.array(
                [
                    [math.cos(self._angle), math.sin(self._angle)],
                    [math.sin(-self._angle), math.cos(self._angle)],
                ]
            )
            measurements = self._measurement_mat.reshape((self._n_boxes, 2)) @ rot_mat

            if timestamp is not None:
                duration = timestamp - self._timestamp
                assert (
                    0 < duration < 5
                ), f"Long duration! {timestamp=} {self._timestamp=}"
                self._timestamp = timestamp
            else:
                duration = force_duration
                self._timestamp += force_duration

            self._update_camera(measurements, duration)

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

    def _advance(self, timestamp: float):
        assert timestamp > self._timestamp, "Timestamp has already passed!"
        duration = timestamp - self._timestamp

        self._timestamp = timestamp
        self._transition_mat[1:, 0] = np.tile([0, -duration], self._n_boxes)
        sigma_x = self._transition_covar * duration

        self._cur_mean = self._transition_mat @ self._cur_mean
        self._cur_covar = (
            self._transition_mat @ self._cur_covar @ self._transition_mat.T + sigma_x
        )

    def update_movement(self, move: MovementAction, advance=True):
        assert move.timestamp - self._timestamp > -0.01, "Out of order move update!"
        with self._lock:
            if advance and move.timestamp > self._timestamp:
                self._advance(move.timestamp)

            self._cur_mean[0] = move.speed
            self._cur_covar[0, 0] = 10 if move.confident else 200

    def state(self, permissible_variance: float = 200.0) -> StateEstimate:
        with self._lock:
            boxes = []
            var_diag = np.diag(self._cur_covar)
            for box in range(self._n_boxes):
                if var_diag[1 + box * 2 : 3 + box * 2].max() < permissible_variance:
                    boxes.append(
                        Box(box + 1, *self._cur_mean[1 + box * 2 : 3 + box * 2])
                    )

            return StateEstimate(Box(0, *self._cur_mean[:2]), boxes)


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
        print("State:      ", state.state())

    print("True y:     ", coords[:, 1] + speeds[-1])
    print(f"took {time.perf_counter() - start} seconds")


if __name__ == "__main__":
    main_test()
