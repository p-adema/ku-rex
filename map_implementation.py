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


class GridOccupancyMap:
    """ """

    def __init__(self, low=(0, 0), high=(2, 2), res=0.05) -> None:
        self.map_area = [low, high]  # a rectangular area
        self.map_size = np.array([high[0] - low[0], high[1] - low[1]])
        self.resolution = res

        self.n_grids = [int(s // res) for s in self.map_size]

        self.grid = np.zeros((self.n_grids[0], self.n_grids[1]), dtype=np.uint8)

        self.extent = [
            self.map_area[0][0],
            self.map_area[1][0],
            self.map_area[0][1],
            self.map_area[1][1],
        ]

    def in_collision(self, pos):
        """
        find if the position is occupied or not. return if the queried pos is outside the map
        """
        indices = [
            int((pos[i] - self.map_area[0][i]) // self.resolution) for i in range(2)
        ]
        for i, ind in enumerate(indices):
            if ind < 0 or ind >= self.n_grids[i]:
                return 1

        return self.grid[indices[0], indices[1]]

    def populate(self, n_obs=6):
        """
        generate a grid map with some circle shaped obstacles
        """
        origins = np.random.uniform(
            low=self.map_area[0] + self.map_size[0] * 0.2,
            high=self.map_area[0] + self.map_size[0] * 0.8,
            size=(n_obs, 2),
        )
        radius = np.random.uniform(low=0.1, high=0.3, size=n_obs)
        # fill the grids by checking if the grid centroid is in any of the circle
        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array(
                    [
                        self.map_area[0][0] + self.resolution * (i + 0.5),
                        self.map_area[0][1] + self.resolution * (j + 0.5),
                    ]
                )
                for o, r in zip(origins, radius):
                    if np.linalg.norm(centroid - o) <= r:
                        self.grid[i, j] = 1
                        break

    # Objects are made up of (x, y) coordinates.
    # Overestimates size of objects and assumes all objects are of roughly same size.
    def add_objects(self, obj_list, obj_scale, size):
        s_dif = np.array(obj_scale) - self.map_size
        sd_objs = np.array(obj_list) / s_dif

        for i in range(self.n_grids[0]):
            for j in range(self.n_grids[1]):
                centroid = np.array(
                    [
                        self.map_area[0][0] + self.resolution * (i + 0.5),
                        self.map_area[0][1] + self.resolution * (j + 0.5),
                    ]
                )
                for o in sd_objs:
                    if np.linalg.norm(centroid - o) <= max(
                        size / s_dif[0], size / s_dif[1]
                    ):
                        self.grid[i, j] = 1
                        break

    def draw_map(self):
        # note the x-y axes difference between imshow and plot
        plt.imshow(
            self.grid.T,
            cmap="Greys",
            origin="lower",
            vmin=0,
            vmax=1,
            extent=self.extent,
            interpolation="none",
        )


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
    lm.draw_map()
    plt.show()
