from typing import Protocol

import numpy as np
from numpy import ndarray


class MovementPredictor(Protocol):
    def predict(self, t1: float, t2: float) -> tuple[np.ndarray, float]: ...


class LinearInterpolation:
    def __init__(
        self,
        point_1: np.ndarray,
        point_2: np.ndarray,
        start_time: float,
        duration: float,
    ):
        self.direction = point_2 - point_1
        self.duration = duration
        self.start_time = start_time

    def _predict_1(self, t: float) -> np.ndarray:
        scale = np.clip((t - self.start_time) / self.duration, 0, 1)
        return self.direction * scale

    def predict(self, t1: float, t2: float) -> tuple[ndarray, float]:
        return self._predict_1(t2) - self._predict_1(t1), 0.0


class LinearTurn:
    def __init__(self, angle: float, start_time: float, duration: float):
        self.angle = angle
        self.start_time = start_time
        self.duration = duration

    def _predict_1(self, t: float) -> float:
        scale = np.clip((t - self.start_time) / self.duration, 0, 1)
        return self.angle * scale

    def predict(self, t1: float, t2: float) -> tuple[ndarray, float]:
        return np.array([0, 0]), (self._predict_1(t2) - self._predict_1(t1))


class Stopped:
    def predict(self, t1: float, t2: float) -> tuple[ndarray, float]:
        return np.array([0, 0]), 0.0
