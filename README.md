# Robotic Experiments 2024: Arlo Sei race code

_Group: Peter Adema, Rasmus Bak, Oscar Koch-Müller, Emil Vedel Thage
Emil Vedel Thage_

As part of the Robotic Experiments course at the University of Copenhagen, we wrote a program for the Arlo robot system
that can complete a race between four landmark boxes while avoiding obstacles. The robot uses two wheels on the side to
drive while using a camera and three sonar sensors mounted along the front to determine the positions of landmarks and
obstacles. Our solution relies on two main components: a path-planning part that uses landmark positions to gain vision
of the target landmark and a direct approach component that ensures the robot moves within the required distance (40cm)
of the goal when line-of-sight is achieved.

The path-planning component primarily uses the camera to take pictures of landmarks and obstacles. These are identified
using an ArUco code (similar to a QR code) on the side, which provides the robot with an estimate of the relative
positions of boxes. These positions are then fed into an Extended Kalman Filter, which estimates the state of the race.
An RRT-based path-planning algorithm then uses this state to generate an approach plan for the target box. This plan is
then executed using the calibrations for the specific robot we worked on (Arlo Sei) until we spot the target box.

Upon gaining line-of-sight on the target box, the program transitions into direct-approach mode, composed of two phases.
In the first phase, the target box is spotted again, and the robot faces the target directly. If the target is far away,
the robot also approaches, and if there is an obstacle between the target and the robot, the robot moves to the side to
avoid the obstacle. When the robot is correctly angled and within 1.70m, it transitions into the second phase, moving
forward a fixed amount (interrupted if the sonars detect that the robot has already arrived at the box). If this final
approach was successful, the landmark is considered visited, and the process repeats with path-planning towards the next
target.

The code primarily uses two threads, a 'main' thread responsible for most movements, and a 'state' thread responsible
for all sensor measurements and the direct-approach

Race code:

- [aruco_utils.py](aruco_utils.py): Utility functions to capture images and parse ArUco codes
- [box_types.py](box_types.py): Internal types for representing landmarks and obstacles as boxes
- [constants.py](constants.py): Constants for path planning and camera calibration
- [global_state.py](global_state.py): Global state shared between threads
- [kalman_state_fixed.py](kalman_state_fixed.py): An Extended Kalman Filter implementation for retaining state regarding
  boxes and the robot
- [main.py](main.py): Entry point for the race code
- [map_implementation.py](map_implementation.py): Collision detection for path planning
- [move_calibrated.py](move_calibrated.py): Calibrated movement code for precise plan-following
- [movement_predictors.py](movement_predictors.py): Utility types for the state to keep track of movements
- [path_plan.py](path_plan.py): Primary movement code for the main thread
- [robot.py](robot.py): Connection with the microcontroller to send commands (slightly modified from provided code)
- [rrt_landmarks.py](rrt_landmarks.py): RRT implementation for path-planning
- [state_thread.py](state_thread.py): Primary sensor code for running on the state thread

Other:

- [calibration.py](calibration.py): Simple script to turn 180 degrees, use to check battery levels
- [client_link.py](client_link.py)
- [listen_map.py](listen_map.py)