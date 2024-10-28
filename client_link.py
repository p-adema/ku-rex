from __future__ import annotations

import socket
import struct

import numpy as np

from box_types import Box, StateEstimate


class Link:
    def __init__(self, ip, port):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((ip, port))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.s.close()

    def empty(self):
        self.s.send(b"!empty!end")

    def send(self, boxes: list[Box], est_state: StateEstimate, plan: np.ndarray | None, goal: np.ndarray):
        msg = []
        for box in boxes:
            msg.append(box.pack())
        msg.append(b"!state")

        msg.append(struct.pack("<d", est_state.angle))

        msg.append(est_state.robot.pack())

        for s_box in est_state.boxes:
            msg.append(s_box.pack())

        msg.append(b"!path")

        if plan is not None:
            path_flat = plan.flatten()
            msg.append(struct.pack(f"<{len(path_flat)}d", *path_flat))

        msg.append(b"!goal")
        msg.append(struct.pack('<dd', goal[0], goal[1]))
        msg.append(b"!end")
        self.s.send(b"".join(msg))
        msg.clear()
