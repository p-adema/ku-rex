import robot

arlo = robot.Robot()
compass = int(input("Comapss? "))
# 115 deg / s
ms: dict[int, int] = {}
try:
    while True:
        turn_time = float(input("Turn time:"))
        arlo.go(41, -40, t=turn_time)
        arlo.stop()
        new_compass = int(input("Compass?"))
        ms[turn_time] = new_compass - compass % 360
        compass = new_compass
except KeyboardInterrupt:
    arlo.stop()
    print(ms)

# arlo.go(66, 64)
# while arlo.read_front_ping_sensor() > 400:
#     pass
