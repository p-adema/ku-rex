import time

import robot

arlo = robot.Robot()

arlo.go_diff(66, 63, False, False)
time.sleep(2)
arlo.stop()
time.sleep(0.1)


arlo.go_diff(66, 63, True, True)
while arlo.read_front_ping_sensor() > 700:
    time.sleep(1 / 20)

speed = 66
while arlo.read_front_ping_sensor() > 427:
    changed = False
    if speed > 36:
        speed -= 6
        changed = True

    if changed:
        time.sleep(1 / 40)
        arlo.go_diff(speed, round(speed * 63 / 66), True, True)
    time.sleep(1 / 40 if changed else 1 / 30)

print(arlo.read_front_ping_sensor())
