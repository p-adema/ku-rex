import robot

arlo = robot.Robot()
input("Enter to start")
print("Running ...")
arlo.go(120, 120, t=0.2)
for i in range(10):
    arlo.go(60, 120, t=5.4 if not i else 6.7)
    arlo.go(120, 120, t=0.1)
    arlo.go(120, 50, t=6.8 if not i else 7)
    print(arlo.read_back_ping_sensor())
    print(arlo.read_back_ping_sensor())
    arlo.go(120, 120, t=0.15)
    print(arlo.read_back_ping_sensor())
    print(arlo.read_back_ping_sensor())
    print()

arlo.stop()
