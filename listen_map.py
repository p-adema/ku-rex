import socket
from typing import NamedTuple

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from aruco_utils import server_ip, server_port
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
    cam: list[Box]
    state: list[Box]
    visible: set[int]


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

        cam_b, state_b = update.split(b"!state")

        cam = Box.unpack_multi(cam_b)
        state = Box.unpack_multi(state_b)
        visible = {box.id for box in cam}

        updates.append(Observation(cam, state, visible))

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

        for obs in updates:
            state.update_camera(obs.boxes)
            est = state.state()
            # assert est.boxes == obs.state, "State divergence!"

            print(f"Going {est.speed} mm/s")
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
        ax_state.add_artist(
            Circle(
                (box.x, box.y),
                200,
                fill=False,
                color="green" if box.id in update.visible else "red",
            )
        )
        ax_state.text(box.x - 70, box.y - 70, str(box.id), ma="center", s=10)


def main():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        server.bind((server_ip, server_port))
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
