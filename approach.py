import time

import robot

FAR_TARGET = 1_000
RATIO_TARGET = 0.87
FRONT_TARGET = 420
RIGHT_TARGET = 2070
BACK_TARGET = 2048

POWER_LEFT = 66
POWER_RIGHT = 63

arlo = robot.Robot()


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
    if RATIO_TARGET - 0.5 < ratio < RATIO_TARGET + 0.5:
        pass
    elif ratio > RATIO_TARGET:
        # turn right
        print("right")
        crawl(lambda: sample_ratio() < RATIO_TARGET + 0.5, left=33, right=-31)
    else:
        # turn left
        print("left")
        crawl(lambda: sample_ratio() > RATIO_TARGET - 0.5, left=-33, right=31)

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


approach()
arlo.stop()
time.sleep(0.2)
print(arlo.read_back_ping_sensor())


# arlo.stop()
