import numpy as np

measurements = np.load("data/arc.npy").T

measurements

# lap_1d = np.array([-1, 1])
#
# for name, ms, r in zip(("left", "front", "right"), measurements.T, (0, 0, -0)):
#     ms = np.roll(ms, r)
#     diff = np.correlate(ms, lap_1d, mode="valid")
#     plt.title(name)
#     plt.plot(range(len(diff)), diff, label="diff")
#     plt.plot(range(len(ms)), ms, label="measurements")
#     plt.legend()
#     plt.show()
