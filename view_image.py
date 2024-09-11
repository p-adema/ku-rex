import matplotlib.pyplot as plt
import numpy as np

img = np.load("data/photo_approx_50.npy")
plt.imshow(img[:, :, ::-1])
plt.show()

# 14.3, 14.4 cm shape
