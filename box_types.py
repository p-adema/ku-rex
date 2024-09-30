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
BOX_SIZE = 200
BOX_SIZE_MARGIN = BOX_SIZE + 150


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
    for cbox in observed:
        boxes_dup.setdefault(cbox.id, []).append((cbox.t_vec[0, 0], cbox.t_vec[2, 0]))

    boxes = []
    for name, coords in boxes_dup.items():
        x, y = np.array(coords).mean(0).astype(int)
        y += 200 // len(coords)
        boxes.append(Box(name, x, y))

    return boxes


class StateEstimate(NamedTuple):
    robot: Box
    boxes: list[Box]


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
