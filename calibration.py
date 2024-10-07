from time import sleep

import cv2 as cv
import numpy as np

import robot

# Create a robot object and initialize
arlo = robot.Robot()
input("Enter to start")
print("Running ...")

leftSpeed = 66
rightSpeed = 63
waitTime = 0.041

FAR_TARGET = 1_000
RATIO_TARGET = 0.87
FRONT_TARGET = 420
RIGHT_TARGET = 2070
BACK_TARGET = 2048


def drive():
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
    sleep(2.3 - waitTime)
    # print(arlo.go_diff(leftSpeed/2, 0, 1, 1))
    sleep(waitTime)
    stopDrive()


def stopDrive():
    for i in range(4):
        print(arlo.go_diff(leftSpeed - (10 * i), rightSpeed - (10 * i), 1, 1))
        sleep(0.1)
    print(arlo.stop)


def turnLeft():
    print(arlo.go_diff(leftSpeed, rightSpeed, 0, 1))
    sleep(0.75 - waitTime)
    print(arlo.go_diff(leftSpeed // 2, 0, 1, 1))
    sleep(waitTime)
    print(arlo.stop())


def turnRight():
    print(arlo.go_diff(leftSpeed, rightSpeed, 1, 0))
    sleep(0.7)
    print(arlo.stop())


def crawl(stop, left: int, right: int):
    arlo.go(left, right)
    while not stop():
        pass


def crawl_forward(target: int):
    crawl(lambda: arlo.read_front_ping_sensor() < target + 300, left=66, right=63)
    crawl(lambda: arlo.read_front_ping_sensor() < target + 100, left=45, right=43)
    crawl(lambda: arlo.read_front_ping_sensor() < target + 20, left=33, right=31)


def sample_ratio() -> float:
    left = sorted(arlo.read_left_ping_sensor() for _ in range(5))[2]
    right = sorted(arlo.read_right_ping_sensor() for _ in range(5))[2]
    return left / right


def approach():
    crawl_forward(FAR_TARGET)
    ratio = sample_ratio()
    print(f"{ratio=}")
    if RATIO_TARGET - 0.3 < ratio < RATIO_TARGET + 0.3:
        pass
    elif ratio > RATIO_TARGET:
        # turn right
        print("right")
        crawl(lambda: sample_ratio() < RATIO_TARGET + 0.3, left=33, right=-31)
    else:
        # turn left
        print("left")
        crawl(lambda: sample_ratio() > RATIO_TARGET - 0.3, left=-33, right=31)

    crawl_forward(FRONT_TARGET)

    arlo.go(-64, +62, t=0.78)

    measurement = arlo.read_back_ping_sensor()
    if measurement > BACK_TARGET:
        crawl(
            lambda: arlo.read_back_ping_sensor() - BACK_TARGET < 8, left=-31, right=-31
        )
    else:
        crawl(
            lambda: RIGHT_TARGET - arlo.read_back_ping_sensor() < 8, left=31, right=31
        )


FRONT_TARGET = 420
RIGHT_TARGET = 2070
BACK_TARGET = 2048

"""approach()
for _ in range(5):
    for _ in range(3):
        drive()
        sleep(0.5)
        # print("Left = " + str(arlo.read_left_wheel_encoder()))
        # print("Right = " + str(arlo.read_right_wheel_encoder()))
        turnLeft()
        sleep(0.5)

    approach()
    sleep(0.5)

arlo.stop()"""
# sleep(1)
# drive()
# sleep(0.5)
# print(arlo.reset_encoder_counts())
#
# sleep(waitTime)
# print(arlo.go_diff(leftSpeed, rightSpeed, 1, 1))
# sleep(1)
# print(arlo.stop())
# sleep(waitTime)
# print("Left = " + str(arlo.read_left_wheel_encoder()))
# sleep(waitTime)
# print("Right = " + str(arlo.read_right_wheel_encoder()))

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

import aruco_utils

cam = aruco_utils.get_camera_picamera()

images = []
for i in range(10):
    print("Say chess")
    image = cam.capture_array("main")[::, ::]
    np.save(f"data/CalPic_{i}", image)
    images.append(image)
    for i in range(4):
        print(f"{i+1}")
        sleep(1)

"""for img in images:
    print('New Image')
    
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,6), None)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
 
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
 
cv.destroyAllWindows()

print(objpoints)
print(imgpoints)
#ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)"""
