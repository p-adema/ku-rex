import time

import numpy as np

from box_types import Box


class KalmanState:
    def __init__(self, n_boxes: int = 4):
        self._n_boxes = n_boxes
        self._f_mat = np.identity(1 + n_boxes * 2)
        self._sigma_x = np.diag([100] + [0.1, 0.1] * n_boxes)
        self._u_t = np.zeros((1 + n_boxes * 2,))
        self._sigma_t = np.diag([10] + [1_000_000, 1_000_000] * n_boxes)
        self._h_mat = np.hstack((np.zeros((2 * n_boxes, 1)), np.identity(2 * n_boxes)))
        self._sigma_z = np.diag(np.repeat(20, n_boxes * 2))
        self._state_identity = np.eye(1 + self._n_boxes * 2)
        self._last_t = None

    def update_camera(self, boxes: list[Box], force_duration=None):
        assert len(set(b.id for b in boxes)) == len(boxes), f"Duplicate: {boxes}"
        assert max(b.id for b in boxes) <= self._n_boxes, f"High ID: {boxes}"
        assert min(b.id for b in boxes) > 0, f"Low ID: {boxes}"
        assert max(max(abs(b.x), abs(b.y)) for b in boxes) < 10_000, f"OOB: {boxes}"

        measurement = np.zeros((self._n_boxes * 2,))
        update_mask = np.zeros((self._n_boxes * 2,))

        for box in boxes:
            measurement[(box.id - 1) * 2 : box.id * 2] = box.x, box.y
            update_mask[(box.id - 1) * 2 : box.id * 2] = 1
        np.fill_diagonal(self._h_mat[:, 1:], update_mask)

        if self._last_t is not None and force_duration is None:
            duration = time.perf_counter() - self._last_t
        else:
            duration = force_duration or 0.5

        self._update_camera(measurement, duration)

    def _update_camera(self, measurement: np.ndarray, duration_sec: float):
        assert measurement.shape == (self._n_boxes * 2,)
        assert 0 < duration_sec < 5

        self._f_mat[1:, 0] = np.tile([0, -duration_sec], self._n_boxes)
        sigma_x = self._sigma_x * duration_sec

        pred_sig = self._f_mat @ self._sigma_t @ self._f_mat.T + sigma_x
        pred_half_update = pred_sig @ self._h_mat.T

        gain = pred_half_update @ np.linalg.inv(
            self._h_mat @ pred_half_update + self._sigma_z
        )
        pred_u = self._f_mat @ self._u_t

        self._u_t = pred_u + gain @ (measurement - self._h_mat @ pred_u)
        self._sigma_t = (self._state_identity - gain @ self._h_mat) @ pred_sig
        self._last_t = time.perf_counter()

    def state(self, permissible_variance: float = 200.0) -> tuple[int, list[Box]]:
        boxes = []
        var_diag = np.diag(self._sigma_t)
        for box in range(self._n_boxes):
            if var_diag[1 + box * 2 : 3 + box * 2].max() < permissible_variance:
                boxes.append(Box(box + 1, *self._u_t[1 + box * 2 : 3 + box * 2]))

        return int(self._u_t[0]), boxes

    def debug_print(self):
        diag = [
            num if num < 10_000 else "high"
            for num in np.round(np.diag(self._sigma_t)).astype(int)
        ]
        print(
            f"think I'm moving with speed {self._u_t[0]: <6.1f}±{diag[0]: >4}, seeing",
            " and ".join(
                f"{box+1}"
                f"({self._u_t[1 + box * 2]: <5.0f}±{diag[1 + box * 2]: >4},"
                f" {self._u_t[2 + box * 2]: <5.0f}±{diag[2 + box * 2]: >4})"
                for box in range(self._n_boxes)
            ),
        )


def main_test():
    rng = np.random.default_rng(seed=1)
    coords = np.array([[100.0, 100], [200, 200]])

    num_measurements = 11
    validities = [(True, True)] * num_measurements
    speeds = np.cumsum(np.repeat(5, num_measurements))
    noise = 10

    state = KalmanState(2)
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
