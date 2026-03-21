import importlib.util
import sys
import time
from pathlib import Path


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_BUS_SERVO_DIR = Path(__file__).resolve().parent / "bus_servo"
_protocol = _load_module("protocol", _BUS_SERVO_DIR / "protocol.py")
_transport = _load_module("transport", _BUS_SERVO_DIR / "transport.py")
_driver = _load_module("driver", _BUS_SERVO_DIR / "driver.py")

SerialTransport = _transport.SerialTransport
BusServoDriver = _driver.BusServoDriver


class BusServo:
    def __init__(self, port="/dev/ttyAMA0", baudrate=9600, servo_num=2, servo_ids=None, moving_time=50):
        self.servo_num = int(servo_num)
        self.servo_ids = list(servo_ids) if servo_ids is not None else [1, 2]
        self.moving_time = int(moving_time)
        self.transport = SerialTransport(port=port, baudrate=baudrate, timeout=1.0)
        self.driver = BusServoDriver(self.transport)
        self.servo_angles_setting = [[sid, 500] for sid in self.servo_ids]
        self.servo_angles_reading = [[sid, 500] for sid in self.servo_ids]
        self.pan_min_angle = -12.0
        self.pan_max_angle = 12.0
        self.pan_min_angle_index = 450
        self.pan_max_angle_index = 550
        self.tilt_min_angle = -12.0
        self.tilt_max_angle = 12.0
        self.tilt_min_angle_index = 450
        self.tilt_max_angle_index = 550
        self.pan_bound_flag = False
        self.tilt_bound_flag = False
        self.angle_equal_threshold = 2
        self.reset()

    def set_angle(self, servo_id, angle):
        if servo_id not in self.servo_ids:
            return None
        angle = float(angle)
        if servo_id == 1:
            self.pan_bound_flag = False
            angle = self._clip_pan(angle)
            angle_index = self.map(
                angle,
                self.pan_min_angle,
                self.pan_max_angle,
                self.pan_min_angle_index,
                self.pan_max_angle_index,
            )
            self._update_setting(servo_id, angle_index)
            return self.pan_bound_flag
        if servo_id == 2:
            self.tilt_bound_flag = False
            angle = self._clip_tilt(angle)
            angle_index = self.map(
                angle,
                self.tilt_min_angle,
                self.tilt_max_angle,
                self.tilt_min_angle_index,
                self.tilt_max_angle_index,
            )
            self._update_setting(servo_id, angle_index)
            return self.tilt_bound_flag
        angle_index = max(0, min(1000, int(round(angle))))
        self._update_setting(servo_id, angle_index)
        return False

    def set_angles(self, angles):
        for servo_id, angle in angles:
            self.set_angle(servo_id, angle)

    def move_angle(self, wait=True):
        for servo_id, angle_index in self.servo_angles_setting:
            self.driver.move_time_write(servo_id, int(angle_index), self.moving_time)
        if wait:
            time.sleep(self.moving_time / 1000.0)

    def read_servos_angle(self):
        for i, servo_id in enumerate(self.servo_ids):
            angle = int(self.driver.read_pos(servo_id))
            self.servo_angles_reading[i] = [servo_id, angle]
        return self.servo_angles_reading

    def are_lists_soft_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False
        for i in range(len(list1)):
            if abs(int(list1[i][1]) - int(list2[i][1])) > self.angle_equal_threshold:
                return False
        return True

    def map(self, x, in_min, in_max, out_min, out_max):
        return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)

    def reset(self):
        if 1 in self.servo_ids:
            self.set_angle(1, 0.0)
        if 2 in self.servo_ids:
            self.set_angle(2, 0.0)
        self.move_angle()

    def cleanup(self):
        self.driver.close()

    def _update_setting(self, servo_id, angle_index):
        for i, (sid, _) in enumerate(self.servo_angles_setting):
            if sid == servo_id:
                self.servo_angles_setting[i][1] = int(angle_index)
                return
        self.servo_angles_setting.append([servo_id, int(angle_index)])

    def _clip_pan(self, angle):
        if angle > self.pan_max_angle:
            self.pan_bound_flag = True
            return self.pan_max_angle
        if angle < self.pan_min_angle:
            self.pan_bound_flag = True
            return self.pan_min_angle
        return angle

    def _clip_tilt(self, angle):
        if angle > self.tilt_max_angle:
            self.tilt_bound_flag = True
            return self.tilt_max_angle
        if angle < self.tilt_min_angle:
            self.tilt_bound_flag = True
            return self.tilt_min_angle
        return angle
