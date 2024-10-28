import math
import socket
import struct
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

import constants
from kalman_state import Box


def axes_base(ax_live: plt.Axes, ax_state: plt.Axes):
    for ax in (ax_live, ax_state):
        ax.clear()
        ax.set_xlabel("Width")
        ax.set_ylabel("Depth")
        ax.set_aspect("equal")

    ax_live.scatter(0, 225, marker="o", c="red", s=10)
    ax_live.add_artist(Circle((0, 0), 225, fill=True, color="black"))

    ax_live.set_ylim((-500, 6_000))
    ax_live.set_xlim((-2000, 2000))
    ax_state.set_xlim((constants.map_min_x, constants.map_max_x))
    ax_state.set_ylim((constants.map_min_y, constants.map_max_y))
    plt.tight_layout()


class Observation(NamedTuple):
    cam: list[Box]
    state: list[Box]
    visible: set[int]
    path: np.ndarray
    goal: np.ndarray
    angle: float


def parse_msg(msg: bytes) -> tuple[bool, list[Observation]]:
    closed = False
    if not msg or msg[-6:] == b"!close":
        closed = True
        msg = msg.rstrip(b"!close")

    updates = []

    for update in msg.split(b"!end"):
        update = update.rstrip(b"!empty")
        if not update:
            continue

        cam_b, state_path = update.split(b"!state")
        angle_state_b, path__goal_b = state_path.split(b"!path")
        path_b, goal_b = path__goal_b.split(b"!goal")

        cam = Box.unpack_multi(cam_b)
        angle = struct.unpack("<d", angle_state_b[:8])[0]
        state = Box.unpack_multi(angle_state_b[8:])
        path = np.array(struct.unpack(f"<{len(path_b) // 8}d", path_b)).reshape((-1, 2))
        goal = np.array(struct.unpack("<dd", goal_b))
        visible = {box.id for box in cam}

        updates.append(Observation(cam, state, visible, path, goal, angle))

    return closed, updates


def handle_robot(client: socket.socket):
    plt.ion()
    fig, (ax_live, ax_state) = plt.subplots(ncols=2)
    fig: plt.Figure
    axes_base(ax_live, ax_state)
    plt.pause(0.1)
    closed = False
    # state = KalmanState(n_boxes=4)

    while not closed:
        msg = client.recv(4096)
        closed, updates = parse_msg(msg)
        plt.pause(0.01)

        for obs in updates:
            # state.update_camera(obs.cam, force_duration=0.2)
            # est = state.state()
            # assert est.boxes == obs.state, "State divergence!"

            # print(f"Going {est.speed} mm/s")
            axes_base(ax_live, ax_state)
            axes_live(ax_live, obs)
            axes_state(ax_state, obs)
            plt.pause(0.01)

    plt.ioff()
    plt.close("all")
    plt.pause(0.01)


def axes_live(ax_live: plt.Axes, update: Observation):
    for box in update.cam:
        ax_live.add_artist(Circle((box.x, box.y), 200, fill=False, color="green"))
        ax_live.text(box.x - 70, box.y - 70, str(box.id), ma="center")


def axes_state(ax_state: plt.Axes, update: Observation):
    for box in update.state:
        if box.id != 0:
            ax_state.add_artist(
                Circle(
                    (box.x, box.y),
                    200,
                    fill=False,
                    color="green" if box.id in update.visible else "red",
                )
            )
            ax_state.add_artist(
                Circle(
                    (box.x, box.y),
                    constants.box_size_margin,
                    fill=False,
                    color="blue",
                    linestyle="dashed",
                )
            )
            ax_state.text(box.x - 70, box.y - 70, str(box.id), ma="center")
        else:
            # print("angle", math.degrees(update.angle))
            rot_mat = np.array(
                [
                    [math.cos(update.angle), math.sin(-update.angle)],
                    [math.sin(update.angle), math.cos(update.angle)],
                ]
            )
            camera_pos = (rot_mat @ np.array([[225], [0]])).flatten() + box
            ax_state.scatter(camera_pos[0], camera_pos[1], marker="o", c="red", s=10)
            ax_state.add_artist(Circle((box.x, box.y), 225, fill=True, color="black"))
    ax_state.plot(update.path[:, 0], update.path[:, 1])
    ax_state.scatter(update.goal[0], update.goal[1], s=20, marker="x", c="red")


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((constants.server_ip, constants.server_port))
        server.listen(0)
        while True:
            s, addr = server.accept()
            print(f"Accepted connection from {addr[0]}:{addr[1]}")
            handle_robot(s)
    except KeyboardInterrupt:
        print("Shutting down...")
    except ConnectionError as e:
        print(f"Connection failed: {e}")
    finally:
        server.close()


if __name__ == "__main__":
    main()
