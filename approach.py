import time

import robot

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


crawl(lambda: arlo.read_front_ping_sensor() < FRONT_TARGET + 300, left=66, right=63)
crawl(lambda: arlo.read_front_ping_sensor() < FRONT_TARGET + 100, left=45, right=43)
crawl(lambda: arlo.read_front_ping_sensor() < FRONT_TARGET + 20, left=33, right=31)
arlo.go(-64, +62, t=0.78)

measurement = arlo.read_back_ping_sensor()
if measurement > BACK_TARGET:
    crawl(lambda: arlo.read_back_ping_sensor() - BACK_TARGET < 8, left=-31, right=-31)
else:
    crawl(lambda: RIGHT_TARGET - arlo.read_back_ping_sensor() < 8, left=31, right=31)


arlo.stop()
time.sleep(0.2)
print(arlo.read_back_ping_sensor())

# if arlo.read_right_ping_sensor() > RIGHT_TARGET:
#     crawl(lambda: arlo.read_right_ping_sensor() < RIGHT_TARGET + 20, left=-33, right=31)
# else:
#     crawl(lambda: arlo.read_right_ping_sensor() > RIGHT_TARGET - 20, left=33, right=-31)
#
# crawl(lambda: arlo.read_front_ping_sensor() < 420 + 20, left=33, right=31)
# arlo.stop()
# sys.exit(1)


# arlo.stop()
