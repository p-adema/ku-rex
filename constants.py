from __future__ import annotations

from box_types import Box

marker_size_mm = 144
avg_focal = 416.0  # Full resolution: 1727.9, recalibrated 416.0
img_width = 320
img_height = 180
server_ip = "192.168.51.159"
server_port = 1809

map_min_x = -500
map_max_x = +4_500
map_min_y = -500
map_max_y = +3_500

box_size = 200
box_size_margin = box_size + 450
KNOWN_BOXES = [
    Box(id=1, x=0, y=0),
    Box(id=2, x=0, y=3_000),
    Box(id=3, x=4_000, y=0),
    Box(id=4, x=4_000, y=3_000),
]
N_START_SCANS = 1
AUTO_SCAN_INTERVAL = 1
