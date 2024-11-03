from __future__ import annotations

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
        self._transition_covar = np.diag([5.0, 5.0] + [0] * (n_boxes * 2))

        # We are certain that we start at (0, 0), facing forwards
        # We are very uncertain about the location of the boxes
        self._cur_mean = np.zeros((2 + n_boxes * 2,), dtype=float)
        self._cur_covar = np.diag([0.0, 0.0] + [1e9] * (n_boxes * 2))

        # A matrix of (Measurement) x (State)
        # gets left-multiplied with the state to produce the predicted measurement
        self._measurement_mat = np.zeros((n_boxes * 2, 2 + n_boxes * 2))
        # (Measurement) x (Measurement) covariance matrix.
        # We're fairly certain about the camera measurements
        self._measurement_covar = np.diag(np.repeat(100, n_boxes * 2))

        self._buf_identity = np.identity(2 + self._n_boxes * 2)
        self._buf_measurement = np.zeros((self._n_boxes * 2,))
        self._timestamp = time.time()
        self._lock = threading.Lock()
        self._movements = queue.Queue()
        self._angle = math.radians(90.0)
        self._move: MovementPredictor = Stopped()
        self._known_boxes: list[Box] | None = None

    def transform_known_boxes(self, boxes: list[Box], center_around: int) -> bool:
        with self._lock:
            success, angle, positions, expected_com = self.estimate_known_transform(
                boxes
            )
            if success is not None:
                return success

            rot_mat = self.rotation_matrix(angle)
            self._angle = (self._angle + angle) % (math.pi * 2)
            rot_positions = (positions - expected_com) @ rot_mat + expected_com
            center_box = next(filter(lambda box: box.id == center_around, boxes))
            rot_positions -= rot_positions[center_around] - np.asarray(center_box)
            print(
                f"Transform before= \n{self._cur_mean.reshape((-1, 2))=}"
                f"\nafter={rot_positions}"
            )
            self._cur_mean = rot_positions.flatten()
            self.force_known_positions(boxes)
            self._known_boxes = boxes

            self._cur_covar[0, 0] = 200
            self._cur_covar[1, 1] = 200
            return True

    def estimate_known_transform(
        self, boxes
    ) -> tuple[bool | None, float | None, np.ndarray | None, np.ndarray | None]:
        assert len(boxes) > 0, "Must have at least one box"
        variance = np.diag(self._cur_covar).reshape((-1, 2)).mean(1)
        spotted_boxes = [box for box in boxes if variance[box.id] < 200]
        assert spotted_boxes, "No boxes spotted!"
        expected_com = np.zeros((1, 2))
        true_com = np.zeros((1, 2))
        current_pos = self._cur_mean.reshape((-1, 2))
        for box in spotted_boxes:
            expected_com += np.asarray(box)
            true_com += current_pos[box.id]

        expected_com, true_com = (
            expected_com / len(spotted_boxes),
            true_com / len(spotted_boxes),
        )

        translation = expected_com - true_com
        positions = self._cur_mean.reshape((-1, 2)) + translation
        angle_estimates = []
        for box in spotted_boxes:
            box_current = (positions[box.id] - expected_com).flatten()
            box_expected = (np.asarray(box) - expected_com).flatten()
            angle_estimates.append(
                (
                    np.arctan2(box_expected[1], box_expected[0])
                    - np.arctan2(box_current[1], box_current[0])
                )
                % (math.pi * 2)
            )
        angle_estimates = np.array(angle_estimates)
        if angle_estimates.min() < 1 or angle_estimates.max() > 5:
            angle_estimates = (angle_estimates + np.pi) % (2 * np.pi)
            offset_pi = True
        else:
            offset_pi = False
        if angle_estimates.max() - angle_estimates.min() > 1:
            print(
                f"Transform failed, {angle_estimates=}, {translation=},"
                f"\t{expected_com=}, {true_com=}"
            )

            return False, None, None, None
        angle = np.mean(angle_estimates)
        if offset_pi:
            angle = (angle - np.pi) % (2 * np.pi)
        print(
            f"Angle is ~ {math.degrees(angle):.0f} degrees ({angle_estimates=}, {positions=}, {translation=})"
        )
        return None, angle, positions, expected_com

    def project_goal(self, goal: np.ndarray, known_boxes=None) -> np.ndarray | None:
        with self._lock:
            if known_boxes is not None:
                self._known_boxes = known_boxes

            if self._known_boxes is None:
                return goal

            success, angle, positions = self.estimate_known_transform(self._known_boxes)
            if success is not None:
                return None

            translation = positions[0].flatten() - self._cur_mean[:2].flatten()
            rot_mat = self.rotation_matrix(-angle)
            new_goal = (
                goal.reshape((1, 2)) - translation.reshape((1, 2)) @ rot_mat
            ).flatten()
            print(f"Goal {goal} is moved to {new_goal} ({translation=}, {angle=})")
            return new_goal

    def known_badness(self) -> float:
        if self._known_boxes is None:
            return 0.0
        error = []
        with self._lock:
            for box in self._known_boxes:
                error.append(
                    np.linalg.norm(
                        self._cur_mean[box.id * 2 : (box.id + 1) * 2] - np.asarray(box)
                    )
                )

        return np.mean(error)

    def force_box_uncertainty(self, std_dev: float = 100.0):
        new_diag = np.clip(np.diag(self._cur_covar), std_dev, None)
        new_diag[:2] = 0
        np.fill_diagonal(self._cur_covar, new_diag)

    def force_known_positions(self, boxes: list[Box]):
        positions = self._cur_mean.reshape((-1, 2))
        for box in boxes:
            positions[box.id] = np.asarray(box)
            self._cur_covar[box.id * 2, box.id * 2] = 0
            self._cur_covar[box.id * 2 + 1, box.id * 2 + 1] = 0

    def update_camera(
        self, boxes: list[Box], timestamp: float, ignore_far: bool = True
    ):
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
            pred_move, pred_turn = self._move.predict(self._timestamp, timestamp)

            duration = timestamp - self._timestamp
            assert (
                0 < duration < 50
            ), f"Long duration! {timestamp}-{self._timestamp}={duration:.1f}"
            _, part_turn = self._move.predict(
                self._timestamp, max(timestamp - 0.02, self._timestamp)
            )
            angle = self._angle + pred_turn - math.radians(90)  # can also be part_turn
            self._angle += pred_turn
            self._cur_mean[:2] += pred_move
            self._timestamp = timestamp

            rot_mat = self.rotation_matrix(angle)
            positions = self._cur_mean.reshape((-1, 2))
            uncertainties = self._cur_covar.reshape((-1, 2))
            for box in boxes:
                box_arr = np.array([box.x, box.y - 225]).reshape((1, 2))
                rot_box = box_arr @ rot_mat
                if (
                    ignore_far
                    and (np.linalg.norm(positions[box.id] - rot_box) > 25000000)
                    and (np.mean(uncertainties[box.id]) < 100)
                ):
                    print(f"Skipping box {box.id}, would have been at {rot_box}")
                    continue

                self._buf_measurement[(box.id - 1) * 2 : box.id * 2] = rot_box
                # Only valid measurements should be taken into account
                self._measurement_mat[-2 + box.id * 2, box.id * 2] = 1
                self._measurement_mat[-1 + box.id * 2, 1 + box.id * 2] = 1
                self._measurement_mat[-2 + box.id * 2, 0] = -1
                self._measurement_mat[-1 + box.id * 2, 1] = -1

            # print(f"{self._measurement_mat.nonzero()=}")
            # print()
            # print(np.diag(self._cur_covar).reshape((-1, 2)).mean(1))

            # if self._known_boxes is not None and not skip_fix:
            #     self.transform_known_boxes(self._known_boxes)

            self._update_camera(self._buf_measurement, duration)

            # if self._known_boxes is not None:
            #     self.force_known_positions(self._known_boxes)

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
        assert 0 < duration < 50, f"Very long duration! ({duration:.2f})"

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

    def set_move_predictor(
        self, move: MovementPredictor, cancelled_at: float = float("inf")
    ):
        with self._lock:
            pred_move, pred_turn = self._move.predict(self._timestamp, cancelled_at)
            self._angle += pred_turn
            self._angle %= np.pi * 2
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

    def propose_movement(self, target: np.ndarray):
        with self._lock:
            print(f"Propose movement: {self._angle=}")
            pos = self._cur_mean[:2].flatten()
            # if angle is None:
            #     angle = self._angle
            diff = target.flatten() - pos
            target_angle = np.arctan2(diff[1], diff[0]) % (np.pi * 2)
            turn = target_angle - self._angle
            print(
                f"Turning {turn}, {math.degrees(turn)}"
                f" (from {math.degrees(self._angle)} to {math.degrees(target_angle)},"
                f" {diff})"
            )

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
