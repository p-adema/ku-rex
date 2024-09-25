import socket
import struct
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from kalman_state import Box, KalmanState


def axes_base(ax_live: plt.Axes, ax_state: plt.Axes):
    for ax in (ax_live, ax_state):
        ax.clear()
        ax.set_xlabel("Width")
        ax.set_ylabel("Depth")
        ax.set_aspect("equal")
        ax.scatter(0, 0, marker="o", c="red", s=10)
        ax.add_artist(Circle((0, -225), 225, fill=True, color="black"))

    ax_live.set_ylim((-500, 6_000))
    ax_live.set_xlim((-2000, 2000))
    ax_state.set_xlim((-5_000, 5_000))
    ax_state.set_ylim((-5_000, 5_000))
    plt.tight_layout()


class Observation(NamedTuple):
    is_moving: bool
    boxes: list[Box]
    visible: set[int]


def parse_msg(msg: bytes) -> tuple[bool, list[Observation]]:
    closed = False
    if not msg or msg[-6:] == b"!close":
        closed = True
        msg = msg.rstrip(b"!close")

    updates = []

    for update in msg.split(b"!end"):
        if not update:
            continue
        update = update.rstrip(b"!empty")
        is_moving = update[-1] == b"m"

        visible = set()
        boxes_dup: dict[int, list[tuple[float, float]]] = {}
        for i in range(len(update) // 17):
            name, x, y = struct.unpack("<Bdd", update[17 * i : 17 * (i + 1)])
            boxes_dup.setdefault(name, []).append((x, y))
            visible.add(name)

        boxes = []
        for name, coords in boxes_dup.items():
            x, y = np.array(coords).mean(0).astype(int)
            y += 200 // len(coords)
            boxes.append(Box(name, x, y))

        updates.append(Observation(is_moving, boxes, visible))

    return closed, updates


def handle_robot(client: socket.socket):
    plt.ion()
    fig, (ax_live, ax_state) = plt.subplots(ncols=2)
    fig: plt.Figure
    axes_base(ax_live, ax_state)
    plt.pause(0.1)
    closed = False
    state = KalmanState(n_boxes=4)

    while not closed:
        msg = client.recv(2048)
        closed, updates = parse_msg(msg)

        for update in updates:
            axes_base(ax_live, ax_state)
            axes_live(ax_live, update)

            state.update_camera(update.boxes)

            s_speed, s_boxes = state.state()
            print(f"Going {s_speed} mm/s ({update.is_moving=})")
            axes_state(ax_state, s_boxes, update)
            plt.pause(0.01)

    plt.ioff()
    plt.close("all")
    plt.pause(0.01)


def axes_state(ax_state, s_boxes, update):
    for box in s_boxes:
        ax_state.add_artist(
            Circle(
                (box.x, box.y),
                200,
                fill=False,
                color="green" if box.id in update.visible else "red",
            )
        )
        ax_state: plt.Axes
        ax_state.text(box.x - 70, box.y - 70, str(box.id), ma="center", s=10)


def axes_live(ax_live: plt.Axes, update):
    for box in update.boxes:
        ax_live.add_artist(Circle((box.x, box.y), 200, fill=False, color="green"))
        ax_live.text(box.x - 70, box.y - 70, str(box.id), ma="center")


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind(("192.168.103.213", 1808))
        server.listen(0)
        while True:
            s, addr = server.accept()
            print(f"Accepted connection from {addr[0]}:{addr[1]}")
            handle_robot(s)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        server.close()


if __name__ == "__main__":
    main()
