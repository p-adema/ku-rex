from __future__ import annotations

import threading

import numpy as np

from box_types import Node
from move_calibrated import CalibratedRobot

CURRENT_GOAL = Node(np.array([0, 0]))
CURRENT_PLAN = None
TARGET_BOX_ID = 1
ALLOW_SPIN_INTERRUPTS = False
SKIP_BOX_MEASUREMENTS = set(range(1, 10))
SONAR_ROBOT_HACK: CalibratedRobot | None = None

stop_program = threading.Event()
turn_ready = threading.Event()
scan_ready = threading.Event()
cancel_spin = threading.Event()
target_line_of_sight = threading.Event()
sonar_aligned = threading.Event()

turn_barrier = threading.Barrier(2)
re_scan_barrier = threading.Barrier(2)
sonar_prep_barrier = threading.Barrier(2)
