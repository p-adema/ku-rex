import time

import numpy as np

d_t = 1 / 20

f_mat = np.array([[1.0, 1.0], [0.0, 1.0]])
sigma_x = np.array([[1.0, 0.5], [0.0, 4.0]])

h_mat = np.array([[1.0, 0.0]])
sigma_z = np.array([10.0])

u_t = np.array([0.0, 0.0])
sigma_t = np.array([[1.0, 1.0], [0.0, 1.0]])

measurements = np.array([0.0, 1.5, 12, 25, 30, 45, 60, 75])
start = time.perf_counter()
for z in measurements:
    # print(f"measured {z}")
    gain = (
        (f_mat @ sigma_t @ f_mat.T + sigma_x)
        @ h_mat.T
        @ np.linalg.inv(
            h_mat @ (f_mat @ sigma_t @ f_mat.T + sigma_x) @ h_mat.T + sigma_z
        )
    )
    u_t = f_mat @ u_t + gain @ (z - h_mat @ f_mat @ u_t)
    sigma_t = (np.eye(2) - gain @ h_mat) @ (f_mat @ sigma_t @ f_mat.T + sigma_x)
    # print(f"think I'm at {u_t[0]:.1f}, moving with speed {u_t[1]:.1f}\n")

print(f"took {time.perf_counter() - start} seconds")
