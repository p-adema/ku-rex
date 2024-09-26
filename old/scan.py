import robot

arlo = robot.Robot()

measurements = {}

try:
    while True:
        dist = int(input("Measured distance (mm): "))
        while True:
            if input(f"Accept {arlo.read_front_ping_sensor()}? ") == "y":
                break
        measurements[dist] = [arlo.read_front_ping_sensor() for _ in range(20)]

except KeyboardInterrupt:
    print("\nDone\n")

print(measurements)
