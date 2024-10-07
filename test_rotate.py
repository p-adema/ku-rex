import time

import tqdm

import robot

arlo = robot.Robot()

test_dur = [
    0.02,
    0.04,
    0.05,
    0.06,
    0.08,
    0.1,
    0.12,
    0.14,
    0.15,
    0.16,
    0.18,
    0.2,
    0.24,
    0.28,
    0.3,
    0.32,
    0.36,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    1.0,
]

results = {}
repetitions = 3


def num(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Not a number!")


deg = num("Starting angle: ")
try:
    progress = tqdm.tqdm(total=len(test_dur) * repetitions)
    for dur in test_dur:
        res = []
        for _ in range(repetitions):
            for _ in range(5):
                arlo.go(+66, -64, t=dur)
                arlo.stop()
                time.sleep(0.5)

            circles = num("Full rotations? ")
            last_deg = num("Ending angle: ")
            turn = last_deg + 360 * circles - deg
            deg = last_deg
            print(f"Turned {turn} degrees.")
            progress.update(1)
            res.append(turn)

        results[dur] = sorted(res)

except KeyboardInterrupt:
    pass
finally:
    print("Turning right results:")
    print(results)
