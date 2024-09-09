from time import sleep

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

approach()
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

arlo.stop()
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
