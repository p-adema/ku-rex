"""

Path planning with Randomized Rapidly-Exploring Random Trees (RRT)

Adapted from
https://github.com/AtsushiSakai/PythonRobotics/blob/master/PathPlanning/RRT/rrt.py
"""

from __future__ import annotations

import timeit

# import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

import constants
from box_types import Box, Node
from map_implementation import DrawExtent, LandmarkMap


class RRT:
    """
    Class for RRT planning
    """

    def __init__(
        self,
        start: Node,
        goal: Node,
        landmarks: LandmarkMap,
        expand_dis: float = 300.0,
        goal_sample_rate: float = 0.3,
        max_iter: int = 500,
        smoothing: bool = True,
    ):
        self.start = start
        self.end = goal
        self.landmarks = landmarks

        self.extend_length = expand_dis
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.smoothing = smoothing
        self._rng = np.random.default_rng(42)

        self.node_list: list[Node] = []
        self.pos_array: np.array = np.repeat(start.pos.reshape((1, 2)), 500, axis=0)

    def plan(self, animation=False, writer=None) -> np.ndarray | None:
        """
        rrt path planning

        animation: flag for animation on or off
        """

        self.node_list = [self.start]

        for _ in range(self.max_iter):
            rng_node = self.get_random_node()
            nearest_ind = self.get_nearest_node_index(rng_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rng_node)

            if self.check_collision_free(new_node):
                self.pos_array[len(self.node_list)] = new_node.pos
                self.node_list.append(new_node)

                if new_node.distance_to(self.end) <= self.extend_length:
                    goal_node = self.steer(new_node, self.end)
                    if self.check_collision_free(goal_node):
                        return self.generate_final_course(goal_node)

            if animation:
                self.draw_graph(rng_node)
                if writer is not None:
                    writer.grab_frame()

        print("Warning: no plan!")
        return None  # cannot find path

    def steer(self, from_node: Node, to_node: Node) -> Node:
        d = from_node.distance_to(to_node)

        if d <= self.extend_length:
            pos = to_node.pos
        else:
            direction = to_node.pos - from_node.pos
            direction /= np.linalg.norm(direction)
            pos = from_node.pos + direction * self.extend_length

        return Node(pos=pos, parent=from_node)

    def generate_final_course(self, goal_node: Node) -> np.ndarray:
        if self.smoothing:
            self.smooth_path(goal_node)

        path = []
        node = goal_node
        while node.parent is not None:
            path.append(node.pos)
            node = node.parent
        path.append(node.pos)
        return np.round(np.array(path))

    def smooth_path(self, goal_node: Node) -> None:
        if goal_node.parent is None or goal_node.parent.parent is None:
            return

        could_smooth = True
        while could_smooth:
            could_smooth = False
            node = goal_node
            grandparent = goal_node.parent.parent
            while grandparent is not None:
                if not self.landmarks.in_collision(node, grandparent):
                    node.parent = grandparent
                    could_smooth = True
                else:
                    node = node.parent

                grandparent = node.parent.parent

    def get_random_node(self) -> Node:
        if self._rng.random() < self.goal_sample_rate:
            near_goal = self._rng.normal(loc=self.end.pos, scale=[500, 500], size=(2,))
            clipped_goal = np.clip(
                near_goal,
                (constants.map_min_x, constants.map_min_y),
                (constants.map_max_x, constants.map_max_y),
            )
            return Node(clipped_goal)
        else:
            return Node(
                np.array(
                    [
                        self._rng.uniform(constants.map_min_x, constants.map_max_x),
                        self._rng.uniform(constants.map_min_y, constants.map_max_y),
                    ],
                    dtype=float,
                )
            )

    # def draw_graph(self, rnd=None) -> None:
    #     # # for stopping simulation with the esc key.
    #     # plt.gcf().canvas.mpl_connect(
    #     #     'key_release_event',
    #     #     lambda event: [exit(0) if event.key == 'escape' else None])
    #     plt.clf()
    #     if rnd is not None:
    #         plt.plot(rnd.pos[0], rnd.pos[1], "^k")
    #
    #     ax = plt.gca()
    #     self.landmarks.draw_map(ax)
    #
    #     for node in self.node_list[1:]:
    #         ax.plot(
    #             (node.pos[0], node.parent.pos[0]),
    #             (node.pos[1], node.parent.pos[1]),
    #             "-g",
    #         )
    #
    #     ax.plot(self.start.pos[0], self.start.pos[1], "xr")
    #     ax.plot(self.end.pos[0], self.end.pos[1], "xr")
    #     plt.pause(0.01)

    def get_nearest_node_index(self, rng_node: Node):
        return np.linalg.norm(
            self.pos_array - rng_node.pos.reshape((1, 2)), axis=1
        ).argmin()

    def check_collision_free(self, node: Node):
        return not self.landmarks.in_collision(node, node.parent)

    @classmethod
    def generate_plan(
        cls, landmarks: list[Box], start: Box, goal: Node, **kwargs
    ) -> np.ndarray | None:
        marks = LandmarkMap(
            landmarks,
            draw_extent=DrawExtent(),
        )

        rrt = RRT(
            start=Node(np.array([start.x, start.y])),
            goal=goal,
            landmarks=marks,
            **kwargs,
        )
        return rrt.plan()


def main():
    landmarks = [
        Box(id=1, x=1_000, y=1_000),
        Box(id=2, x=0, y=1_000),
        Box(id=3, x=100, y=1_400),
    ]

    goal = Node(np.array([0.0, 2_000.0]))
    # rrt = RRT(
    #     start=Node(np.array([0.0, 0.0])),
    #     goal=goal,
    #     landmarks=landmarks,
    #     expand_dis=300,
    #     goal_sample_rate=0.3,
    # )

    # show_animation = False
    metadata = dict(title="RRT Test")
    writer = FFMpegWriter(fps=15, metadata=metadata)
    # fig = plt.figure()

    # with writer.saving(fig, "rrt_test.mp4", 100):
    # path = rrt.plan(animation=show_animation, writer=writer)
    path = RRT.generate_plan(landmarks, goal)

    if path is None:
        print("Cannot find path")
    else:
        print("found path!!")

        # Draw final path
        # if show_animation:
        #     rrt.draw_graph()
        #     plt.plot([x for (x, y) in path], [y for (x, y) in path], "-r")
        #     plt.grid(True)
        #     plt.pause(0.01)  # Need for Mac
        #     plt.show()
        #     writer.grab_frame()
    if path is not None:
        print("One run complete, doing timings...")
        n_runs = 100
        print(
            "Took on average (sec): ",
            timeit.timeit(
                """
    landmarks = [
        Box(id=1, x=1_000, y=1_000),
        Box(id=2, x=0, y=1_000),
        Box(id=3, x=100, y=1_400),
    ]
    goal = Node(np.array([0.0, 2_000.0]))
    path = RRT.generate_plan(landmarks, goal)
    assert path is not None, "Couldn't find path!"
    """,
                globals=globals(),
                number=n_runs,
            )
            / n_runs,
        )


if __name__ == "__main__":
    main()
