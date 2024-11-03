import threading
import time

import numpy as np

import robot
from aruco_utils import get_camera_picamera, sample_markers

bar = threading.Barrier(2)


def cam_thread():
    bar.wait()
    time.sleep(0.1)
    cam = get_camera_picamera(downscale=4, exposure_time_ns=2_000, gain=40)

    print("Cam taking picture!")
    image = cam.capture_buffer().reshape((360 // 2, 640 // 2, 3))
    print("Meta:", cam.capture_metadata())
    np.save("data/photo_moving.npy", image)
    print(image.shape)
    print(sample_markers(image))
    cam.close()


def move_thread():
    arlo = robot.Robot()
    bar.wait()
    arlo.go(-66, 64, t=0.5)
    arlo.stop()
    print("Stopped!")


if __name__ == "__main__":
    t = threading.Thread(target=cam_thread)
    t.start()
    move_thread()
    t.join()
