import robot

arlo = robot.Robot()

measurements = []
# delay = 1 / int(sys.argv[1])
# print(f"delay is {delay}")
# arlo.go_diff(30, 30, True, False)
for _ in range(5):
    print(
        f"{arlo.read_left_ping_sensor()} |"
        f" {arlo.read_front_ping_sensor()} |"
        f" {arlo.read_right_ping_sensor()}"
    )
