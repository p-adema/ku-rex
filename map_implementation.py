"""
Module for interfacing a 2D Map in the form of Grid Occupancy
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from box_types import Box, Node
from constants import box_size_margin


class DrawExtent(NamedTuple):
    x_min: int = -5_000
    x_max: int = 5_000
    y_min: int = -5_000
    y_max: int = 5_000


class LandmarkMap:
    def __init__(
        self,
        landmarks: list[Box],
        draw_extent: DrawExtent,
        changed_radia: dict[int, float],
    ):
        assert landmarks, "No landmarks?"
        self._landmarks = landmarks
        self._distances = np.repeat(box_size_margin, len(landmarks))
        coords = []
        for i, box in enumerate(landmarks):
            coords.append(np.array([box.x, box.y]))
            if box.id in changed_radia:
                self._distances[i] = changed_radia[box.id]

        self._coords: np.ndarray = np.array(coords)
        assert self._coords.shape == (len(landmarks), 2), f"Weird {self._coords.shape=}"
        self._extent = draw_extent
        self._changed_radia = changed_radia

    def in_collision(
        self,
        a: np.ndarray,
        b: np.ndarray,
        relaxed: bool = False,
    ) -> bool:
        if relaxed:
            return self._relaxed_in_collision(a, b)

        a_close = (
            np.linalg.norm(self._coords - a.reshape((1, 2)), axis=1) < self._distances
        )
        if np.any(a_close):
            return True

        b_close = (
            np.linalg.norm(self._coords - b.reshape((1, 2)), axis=1) < self._distances
        )
        if np.any(b_close):
            return True

        direction = b - a
        coords = self._coords - a.reshape((1, 2))
        proj_weight = coords @ direction.reshape((2, 1)) / np.dot(direction, direction)
        within_line = (proj_weight > 0) & (proj_weight < 1)
        proj = proj_weight * direction.reshape((1, 2))
        proj_dist = np.linalg.norm(coords - proj, axis=1)
        proj_close = np.where(
            within_line.reshape((-1, 1)), proj_dist < self._distances, False
        )
        return np.any(proj_close)

    def point_close(self, a: Node) -> bool:
        a_close = (
            np.linalg.norm(self._coords - a.reshape((1, 2)), axis=1) < self._distances
        )
        return np.any(a_close)

    def _relaxed_in_collision(self, a: np.ndarray, b: np.ndarray) -> bool:
        close_boxes_mask = (
            np.linalg.norm(self._coords - a.reshape((1, 2)), axis=1) < self._distances
        ).astype(bool)
        if not np.any(close_boxes_mask):
            print("Relaxed collision ignored")
            self.in_collision(a, b, relaxed=False)

        close_boxes = self._coords[close_boxes_mask]
        print(f"Relaxed collision with {len(close_boxes)=}")

        direction = b - a
        coords = close_boxes - a.reshape((1, 2))
        proj_weight = coords @ direction.reshape((2, 1)) / np.dot(direction, direction)
        if np.any(proj_weight > 0):
            within_line = (proj_weight > 0) & (proj_weight < 1)
            proj = proj_weight * direction.reshape((1, 2))
            proj_dist = np.linalg.norm(coords - proj, axis=1)
            proj_close = np.where(
                within_line.reshape((-1, 1)),
                proj_dist < self._distances - 100,
                False,
            )
            if np.any(proj_close):
                return True

        far_map = LandmarkMap(
            [box for close, box in zip(close_boxes_mask, self._landmarks) if not close],
            self._extent,
            self._changed_radia,
        )
        return far_map.in_collision(a, b)

    def push_back(self, a: Node, b: Node) -> Node:
        pass

    def draw_map(self, axes: plt.Axes):
        axes.clear()
        axes.set_xlabel("Width")
        axes.set_ylabel("Depth")
        axes.set_aspect("equal")
        # ax.scatter(0, 0, marker="o", c="red", s=10)
        # ax.add_artist(Circle((0, -225), 225, fill=True, color="black"))

        axes.set_ylim((self._extent.y_min, self._extent.y_max))
        axes.set_xlim((self._extent.x_min, self._extent.x_max))

        plt.tight_layout()
        for box in self._landmarks:
            axes.add_artist(Circle((box.x, box.y), 200, fill=False, color="green"))
            axes.text(box.x - 70, box.y - 70, str(box.id), ma="center")


if __name__ == "__main__":
    # map = GridOccupancyMap()
    # map.populate()
    #
    # plt.clf()
    # map.draw_map()
    # plt.show()
    lm = LandmarkMap(
        [Box(id=1, x=200, y=200)], draw_extent=DrawExtent(), changed_radia={}
    )
    print(
        lm.in_collision(
            np.array([0.0, -400]),
            np.array([100.0, 100]),
        )
    )
    lm.draw_map(plt.gca())
    plt.show()
