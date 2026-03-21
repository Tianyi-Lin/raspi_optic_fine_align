from __future__ import annotations
from driver import BusServoBoardDriver


class BusServo:
    def __init__(self, driver: BusServoBoardDriver, servo_id: int):
        self.driver = driver
        self.id = servo_id

    def move(self, position: int, time_ms: int):
        self.driver.move_one(self.id, position, time_ms)

    def pos(self) -> int:
        return self.driver.read_one_position(self.id)

    def unload(self):
        self.driver.unload_servos([self.id])