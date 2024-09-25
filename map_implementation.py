"""
Module for interfacing a 2D Map in the form of Grid Occupancy
"""

from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from box_types import BOX_SIZE_MARGIN, Box, Node


class DrawExtent(NamedTuple):
    x_min: int = -5_000
    x_max: int = 5_000
    y_min: int = -5_000
    y_max: int = 5_000


class LandmarkMap:
    def __init__(self, landmarks: list[Box], draw_extent: DrawExtent):
        self._landmarks = landmarks
        self._coords: np.ndarray = np.array([[box.x, box.y] for box in landmarks])
        self._extent = draw_extent

    def in_collision(self, a: Node, b: Node):
        a, b = a.pos, b.pos
        a_close = (
            np.linalg.norm(self._coords - a.reshape((1, 2)), axis=1) < BOX_SIZE_MARGIN
        )
        if np.any(a_close):
            return True

        b_close = (
            np.linalg.norm(self._coords - b.reshape((1, 2)), axis=1) < BOX_SIZE_MARGIN
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
            within_line.reshape((-1, 1)), proj_dist < BOX_SIZE_MARGIN, False
        )
        return np.any(proj_close)

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
    lm = LandmarkMap([Box(id=1, x=200, y=200)], draw_extent=DrawExtent())
    print(
        lm.in_collision(
            Node(np.array([0.0, -400])),
            Node(np.array([100.0, 100])),
        )
    )
    lm.draw_map(plt.gca())
    plt.show()
