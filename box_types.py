from __future__ import annotations

import struct
import time
from typing import NamedTuple

import numpy as np


class CameraBox(NamedTuple):
    id: int
    r_vec: np.ndarray
    t_vec: np.ndarray


_box_fmt = "<Bdd"

CORRECTION_DISTANCES = np.array([70, 198, 208, 270, 320, 540])
CORRECTION_VALUES = np.array([1.72857, 1.29293, 1.22596, 1.19259, 1.16875, 1.088889])


class Box(NamedTuple):
    id: int  # As shown on the box itself
    x: float  # Horizontal displacement w.r.t. the camera, right is positive
    y: float  # Depth displacement w.r.t. the camera, in front is positive

    def __repr__(self):
        return f"Box({self.id}, {self.x:.0f}, {self.y:.0f})"

    def pack(self) -> bytes:
        return struct.pack(_box_fmt, self.id, self.x, self.y)

    @classmethod
    def unpack(cls, b: bytes):
        assert len(b) == 17, f"Invalid byte length ({b!r} length {len(b)})"
        return cls(*struct.unpack(_box_fmt, b))

    @classmethod
    def unpack_multi(cls, b: bytes):
        assert len(b) % 17 == 0, f"Invalid byte length ({b!r} length {len(b)})"
        return [Box.unpack(b[17 * i : 17 * (i + 1)]) for i in range(len(b) // 17)]

    def __array__(self, dtype=None):
        """Numpy interoperability"""
        return np.array((self.x, self.y), dtype=dtype).__array__()


def dedup_camera(observed: list[CameraBox]) -> list[Box]:
    # Can be done better
    boxes_dup: dict[int, list[tuple]] = {}
    # rot = np.empty((3, 3))
    for cbox in observed:
        # cv2.Rodrigues(cbox.r_vec, dst=rot)
        # angle = cv2.RQDecomp3x3(rot)[0][1]
        boxes_dup.setdefault(cbox.id, []).append(
            (
                cbox.t_vec[0, 0],  # + 125 * np.cos(angle),
                cbox.t_vec[2, 0],  # + 125 * np.sin(angle),
            )
        )

    boxes = []
    for name, coords in boxes_dup.items():
        coords = np.asarray(coords).mean(0)
        coords *= np.interp(coords[1], CORRECTION_DISTANCES, CORRECTION_VALUES)
        x, y = coords.astype(int)
        y -= 125 // len(coords) - 500
        boxes.append(Box(name, x, y))

    return boxes


class StateEstimate(NamedTuple):
    robot: Box
    boxes: list[Box]
    angle: float


class MovementAction(NamedTuple):
    speed: float
    timestamp: float
    confident: bool

    @classmethod
    def moving(cls, speed: float, confident: bool = False):
        return cls(speed, timestamp=time.time(), confident=confident)

    @classmethod
    def stop(cls):
        return cls(0, timestamp=time.time(), confident=True)


class Node:
    """
    RRT Node
    """

    __slots__ = ["pos", "parent"]
    __dict__ = None

    def __init__(self, pos: np.ndarray, parent: Node | None = None):
        self.pos: np.ndarray = pos
        self.parent: Node | None = parent

    def distance_to(self, other: Node):
        return np.linalg.norm(other.pos - self.pos)
