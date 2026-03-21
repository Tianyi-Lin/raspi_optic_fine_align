from __future__ import annotations
from driver import BusServoDriver


class BusServo:
    def __init__(self, driver: BusServoDriver, servo_id: int):
        self.driver = driver
        self.id = servo_id

    def move(self, position: int, time_ms: int):
        self.driver.move_time_write(self.id, position, time_ms)

    def move_wait(self, position: int, time_ms: int):
        self.driver.move_time_wait_write(self.id, position, time_ms)

    def start(self):
        self.driver.move_start(self.id)

    def stop(self):
        self.driver.move_stop(self.id)

    def pos(self) -> int:
        return self.driver.read_pos(self.id)

    def vin(self) -> int:
        return self.driver.read_vin(self.id)

    def temp(self) -> int:
        return self.driver.read_temp(self.id)

    def load(self, enable: bool = True):
        self.driver.set_load(self.id, enable)