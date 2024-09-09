# Arlo robot controller
from __future__ import annotations

import time

import serial


class Robot:
    """Defines the Arlo robot API

    DISCLAIMER: This code does not contain error checking - it is the responsibility
    of the caller to ensure proper parameters and not to send commands to the
    Arduino too frequently (give it time to process the command by adding a short sleep
    statement). Failure to do some may lead to strange robot behaviour.

    In case you experience trouble - consider using only commands that do not use the
    encoders.
    """

    def __init__(self, port: str = "/dev/ttyACM0", timeout: int | None = 4) -> None:
        """The constructor port parameter can be changed from default value if you want
        to control the robot directly from your laptop (instead of from the on-board
        raspberry pi). The value of port should point to the USB port on which
        the robot Arduino is connected."""

        print("Waiting for serial port connection ...")
        self.serial = serial.Serial(port, 9600, timeout=timeout)

        while not self.serial.isOpen():
            time.sleep(0.1)
        time.sleep(2)
        print("Running ...")

    def __del__(self) -> None:
        print("Shutting down the robot ...")

        time.sleep(0.05)
        print(self.stop())
        time.sleep(0.1)

        print(self.send_command("k\n"))
        self.serial.close()

    def send_command(self, cmd: str, sleep_ms: float = 0.0) -> bytes:
        """Sends a command to the Arduino robot controller"""
        self.serial.write(cmd.encode("ascii"))
        time.sleep(sleep_ms)
        str_val = self.serial.readline()
        return str_val

    @staticmethod
    def _valid_motor_power(power: int) -> bool:
        """Checks if a power value is in the set {0, [30;127]}.
        This is an internal utility function."""
        return (power == 0) or (40 <= power <= 127)

    def go_diff(
        self, power_left: int, power_right: int, forward_left: bool, forward_right: bool
    ) -> bytes:
        """Change the motor activation

        :param power_left: Left motor power, must zero or in [30, 127]
        :param power_right: Right motor power, must zero or in [30, 127]
        :param forward_left: Left motor direction: on True forward, on False backwards
        :param forward_right: Right motor direction: on True forward, on False backwards

        The Arlo robot may blow a fuse if you run the motors at less than 40 in motor
        power, therefore choose either power = 0 or 30 < power <= 127.
        This does NOT use wheel encoders."""

        assert self._valid_motor_power(power_left), f"Invalid value {power_left=}"
        assert self._valid_motor_power(power_right), f"Invalid value {power_right=}"

        cmd = f"d{power_left:d},{power_right:d},{forward_left:d},{forward_right:d}\n"
        return self.send_command(cmd)

    def go(self, left: int, right: int, t: float = 0.0):
        assert self._valid_motor_power(abs(left)), f"Invalid value {left=}"
        assert self._valid_motor_power(abs(right)), f"Invalid value {right=}"
        cmd = (
            f"d{round(abs(left)):d},{round(abs(right)):d},{left > 0:d},{right > 0:d}\n"
        )
        res = self.send_command(cmd)
        time.sleep(t)
        return res

    def stop(self) -> bytes:
        """Send a stop command to stop motors.
        Sets the motor power on both wheels to zero.
        This does NOT use wheel encoders."""
        return self.send_command("s\n")

    def read_sensor(self, sensor_id: int) -> int | None:
        """Send a read sensor command with sensor_id and return sensor value.
        Returns None on error"""
        res = self.send_command(f"{sensor_id}\n")
        if res:
            return int(res)

    def read_front_ping_sensor(self) -> int:
        """Read the front sonar ping sensor and return the measurement in millimeters"""
        return self.read_sensor(0)

    def read_back_ping_sensor(self) -> int:
        """Read the back sonar ping sensor and return the measurement in millimeters"""
        return self.read_sensor(1)

    def read_left_ping_sensor(self) -> int:
        """Read the left sonar ping sensor and return the measurement in millimeters"""
        return self.read_sensor(2)

    def read_right_ping_sensor(self) -> int:
        """Read the right sonar ping sensor and return the measurement in millimeters"""
        return self.read_sensor(3)

    def read_left_wheel_encoder(self, do_sleep: bool = True) -> bytes:
        """Reads the left wheel encoder counts since last reset_encoder_counts command.
        The encoder has 144 counts for one complete wheel revolution."""
        return self.send_command("e0\n", sleep_ms=0.045 if do_sleep else 0.0)

    def read_right_wheel_encoder(self, do_sleep: bool = True) -> bytes:
        """Reads the right wheel encoder counts since last clear reset_encoder_counts.
        The encoder has 144 counts for one complete wheel revolution."""
        return self.send_command("e1\n", sleep_ms=0.045 if do_sleep else 0.0)

    def reset_encoder_counts(self) -> bytes:
        """Reset the wheel encoder counts."""
        return self.send_command("c\n")
