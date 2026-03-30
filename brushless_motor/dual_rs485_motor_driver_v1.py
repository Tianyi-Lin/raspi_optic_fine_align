#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Raspberry Pi 双 RS485 双电机驱动（第一版）

场景：
- RS485 通道 1 -> 电机 ID=1 -> 左右(Yaw)
- RS485 通道 2 -> 电机 ID=2 -> 俯仰(Pitch)
- 两个电机均工作在“多圈位置模式”
- 0 度视为归中位置
- 两个轴正方向相反，因此分别提供 direction_sign 进行映射

协议依据：
- CMD=0x88: 电机运行
- CMD=0x81: 电机停止
- CMD=0x80: 电机关闭
- CMD=0x9A: 读取状态1
- CMD=0x9C: 读取状态2
- CMD=0x92: 读取多圈角度
- CMD=0xA4: 多圈位置闭环控制命令2（带速度限制）
"""

from __future__ import annotations

import struct
import time
import threading
from dataclasses import dataclass
from typing import Optional, Dict, Any

import serial

try:
    import RPi.GPIO as GPIO  # 树莓派 4 常见
    GPIO_BACKEND = "RPi.GPIO"
except Exception:
    GPIO = None
    GPIO_BACKEND = None

_GPIO_MODE_LOCK = threading.Lock()
_GPIO_PIN_LOCKS: Dict[int, threading.Lock] = {}


def _gpio_ensure_mode_bcm():
    if GPIO is None:
        return
    with _GPIO_MODE_LOCK:
        try:
            if GPIO.getmode() is None:
                GPIO.setmode(GPIO.BCM)
        except Exception:
            GPIO.setmode(GPIO.BCM)


def _gpio_pin_lock(pin: int) -> threading.Lock:
    with _GPIO_MODE_LOCK:
        lock = _GPIO_PIN_LOCKS.get(int(pin))
        if lock is None:
            lock = threading.Lock()
            _GPIO_PIN_LOCKS[int(pin)] = lock
        return lock


class RS485Error(Exception):
    pass


class ChecksumError(RS485Error):
    pass


class ResponseError(RS485Error):
    pass


@dataclass
class MotorConfig:
    name: str
    motor_id: int
    dev: str
    txden_pin: int
    direction_sign: int = 1      # 机械安装方向映射：+1 / -1
    baudrate: int = 1000000
    timeout: float = 0.05
    default_speed_dps: float = 90.0
    min_deg: float = -180.0
    max_deg: float = 180.0


class RS485Port:
    def __init__(self, dev: str, txden_pin: int, baudrate: int = 1000000, timeout: float = 0.05):
        self.dev = dev
        self.txden_pin = txden_pin
        self.baudrate = int(baudrate)
        self.timeout = timeout
        self._rx_buffer = bytearray()
        self._last_flush_ms = 0.0
        self._last_flush_bytes = 0
        self._last_send_ms = 0.0
        self._last_reply_ms = 0.0
        self.ser = serial.Serial(
            port=dev,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            parity=serial.PARITY_NONE,
            stopbits=serial.STOPBITS_ONE,
            timeout=timeout,
        )

        if GPIO is None:
            raise RuntimeError(
                "当前环境无法导入 RPi.GPIO。"
                "若是 Raspberry Pi 5，建议改为 lgpio/gpiozero 版本；"
                "若是 Raspberry Pi OS，请先安装 python3-rpi.gpio。"
            )

        GPIO.setwarnings(False)
        last_err = None
        for _ in range(5):
            try:
                _gpio_ensure_mode_bcm()
                with _gpio_pin_lock(self.txden_pin):
                    GPIO.cleanup(self.txden_pin)
            except Exception:
                pass
            try:
                _gpio_ensure_mode_bcm()
                with _gpio_pin_lock(self.txden_pin):
                    GPIO.setup(self.txden_pin, GPIO.OUT, initial=GPIO.HIGH)
                last_err = None
                break
            except Exception as exc:
                last_err = exc
                try:
                    _gpio_ensure_mode_bcm()
                    with _gpio_pin_lock(self.txden_pin):
                        GPIO.cleanup(self.txden_pin)
                except Exception:
                    pass
                time.sleep(0.05)
        if last_err is not None:
            raise RuntimeError(f"GPIO setup failed for TXDEN pin {self.txden_pin}: {last_err}")

    @staticmethod
    def checksum(data: bytes) -> int:
        return sum(data) & 0xFF

    def _set_send(self) -> None:
        _gpio_ensure_mode_bcm()
        with _gpio_pin_lock(self.txden_pin):
            GPIO.output(self.txden_pin, GPIO.LOW)

    def _set_recv(self) -> None:
        _gpio_ensure_mode_bcm()
        with _gpio_pin_lock(self.txden_pin):
            GPIO.output(self.txden_pin, GPIO.HIGH)

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:
            pass
        try:
            self._set_recv()
        except Exception:
            pass
        try:
            if GPIO is not None:
                _gpio_ensure_mode_bcm()
                with _gpio_pin_lock(self.txden_pin):
                    GPIO.cleanup(self.txden_pin)
        except Exception:
            pass

    def flush_input(self) -> None:
        t0 = time.perf_counter()
        dropped = len(self._rx_buffer)
        self._rx_buffer.clear()
        try:
            while True:
                waiting = int(getattr(self.ser, "in_waiting", 0) or 0)
                if waiting <= 0:
                    break
                chunk = self.ser.read(waiting)
                if not chunk:
                    break
                dropped += len(chunk)
        except Exception:
            self.ser.reset_input_buffer()
        self._last_flush_ms = (time.perf_counter() - t0) * 1000.0
        self._last_flush_bytes = int(dropped)

    def send_frame(self, cmd: int, motor_id: int, payload: bytes = b"") -> None:
        t0 = time.perf_counter()
        header = bytes([0x3E, cmd & 0xFF, motor_id & 0xFF, len(payload) & 0xFF])
        cmd_sum = bytes([self.checksum(header)])
        frame = header + cmd_sum
        if payload:
            frame += payload + bytes([self.checksum(payload)])

        if self._rx_buffer:
            self.flush_input()
        else:
            try:
                if int(getattr(self.ser, "in_waiting", 0) or 0) > 0:
                    self.flush_input()
            except Exception:
                self.flush_input()
        self._set_send()
        time.sleep(0.00025)
        self.ser.write(frame)
        self.ser.flush()
        time.sleep(0.00025)
        self._set_recv()
        self._last_send_ms = (time.perf_counter() - t0) * 1000.0

    def read_exact(self, n: int) -> bytes:
        deadline = time.perf_counter() + max(0.02, self.timeout * 3.0)
        return self._read_exact_until(n, deadline)

    def _fill_rx_buffer_until(self, min_len: int, deadline: float) -> None:
        target_len = max(0, int(min_len))
        while len(self._rx_buffer) < target_len and time.perf_counter() < deadline:
            waiting = 0
            try:
                waiting = int(getattr(self.ser, "in_waiting", 0) or 0)
            except Exception:
                waiting = 0
            to_read = waiting if waiting > 0 else 1
            chunk = self.ser.read(to_read)
            if chunk:
                self._rx_buffer.extend(chunk)

    def _read_exact_until(self, n: int, deadline: float) -> bytes:
        self._fill_rx_buffer_until(n, deadline)
        if len(self._rx_buffer) < n:
            raise ResponseError(f"{self.dev} 读取长度不足: 期望 {n} 字节, 实际 {len(self._rx_buffer)} 字节")
        data = bytes(self._rx_buffer[:n])
        del self._rx_buffer[:n]
        return data

    def _extract_frame_from_buffer(self) -> Optional[bytes]:
        while self._rx_buffer and self._rx_buffer[0] != 0x3E:
            del self._rx_buffer[0]
        if len(self._rx_buffer) < 5:
            return None
        data_len = int(self._rx_buffer[3])
        total_len = 5 + data_len + (1 if data_len > 0 else 0)
        if len(self._rx_buffer) < total_len:
            return None
        frame = bytes(self._rx_buffer[:total_len])
        del self._rx_buffer[:total_len]
        return frame

    def _read_frame_synced(self, deadline: float) -> bytes:
        last_error = None
        while time.perf_counter() < deadline:
            frame = self._extract_frame_from_buffer()
            if frame is not None:
                return frame
            try:
                self._fill_rx_buffer_until(len(self._rx_buffer) + 1, deadline)
            except Exception as exc:
                last_error = exc
        if last_error is not None:
            raise ResponseError(f"{self.dev} 回复读取失败: {last_error}")
        raise ResponseError(f"{self.dev} 回复超时或未找到完整帧")

    def read_reply(self, expected_cmd: int, expected_id: int) -> bytes:
        """
        返回 payload 原始数据（不含 DATA_SUM）
        """
        t0 = time.perf_counter()
        deadline = time.perf_counter() + max(0.02, self.timeout * 3.0)
        last_error = None
        while time.perf_counter() < deadline:
            frame = self._read_frame_synced(deadline)
            header = frame[:5]
            cmd = header[1]
            motor_id = header[2]
            data_len = header[3]
            cmd_sum = header[4]

            if self.checksum(header[:4]) != cmd_sum:
                last_error = ChecksumError(f"{self.dev} 回复 CMD_SUM 校验失败")
                continue

            if data_len == 0:
                if cmd == expected_cmd and motor_id == expected_id:
                    self._last_reply_ms = (time.perf_counter() - t0) * 1000.0
                    return b""
                last_error = ResponseError(
                    f"{self.dev} 收到非目标空回复: 期望 cmd=0x{expected_cmd:02X}/id={expected_id}, 实际 cmd=0x{cmd:02X}/id={motor_id}"
                )
                continue

            payload = frame[5:-1]
            data_sum = frame[-1]
            if self.checksum(payload) != data_sum:
                last_error = ChecksumError(f"{self.dev} 回复 DATA_SUM 校验失败")
                continue

            if cmd != expected_cmd or motor_id != expected_id:
                last_error = ResponseError(
                    f"{self.dev} 收到非目标回复: 期望 cmd=0x{expected_cmd:02X}/id={expected_id}, 实际 cmd=0x{cmd:02X}/id={motor_id}"
                )
                continue

            self._last_reply_ms = (time.perf_counter() - t0) * 1000.0
            return payload

        if last_error is not None:
            raise last_error
        raise ResponseError(f"{self.dev} 回复超时")

    def get_last_io_timing(self) -> Dict[str, float]:
        return {
            "flush_ms": float(self._last_flush_ms),
            "flush_bytes": int(self._last_flush_bytes),
            "send_ms": float(self._last_send_ms),
            "reply_ms": float(self._last_reply_ms),
        }


class LkMotor:
    def __init__(self, cfg: MotorConfig):
        self.cfg = cfg
        self.port = RS485Port(
            dev=cfg.dev,
            txden_pin=cfg.txden_pin,
            baudrate=cfg.baudrate,
            timeout=cfg.timeout,
        )

    @staticmethod
    def _pack_i64(v: int) -> bytes:
        return struct.pack("<q", v)

    @staticmethod
    def _pack_u32(v: int) -> bytes:
        return struct.pack("<I", v)

    @staticmethod
    def _unpack_i64(data: bytes) -> int:
        return struct.unpack("<q", data)[0]

    @staticmethod
    def _unpack_i16(data: bytes) -> int:
        return struct.unpack("<h", data)[0]

    @staticmethod
    def _unpack_u16(data: bytes) -> int:
        return struct.unpack("<H", data)[0]

    def _deg_to_proto(self, deg: float) -> int:
        return int(round(deg * self.cfg.direction_sign * 100.0))

    def _proto_to_deg(self, value: int) -> float:
        return value / 100.0 * self.cfg.direction_sign

    @staticmethod
    def _speed_to_proto(speed_dps: float) -> int:
        if speed_dps < 0:
            raise ValueError("max speed 必须 >= 0")
        return int(round(speed_dps * 100.0))

    def _simple_cmd(self, cmd: int) -> None:
        self.port.send_frame(cmd, self.cfg.motor_id, b"")
        self.port.read_reply(cmd, self.cfg.motor_id)

    def motor_run(self) -> None:
        self._simple_cmd(0x88)

    def motor_stop(self) -> None:
        self._simple_cmd(0x81)

    def motor_off(self) -> None:
        self._simple_cmd(0x80)

    def read_status1(self) -> Dict[str, Any]:
        self.port.send_frame(0x9A, self.cfg.motor_id, b"")
        payload = self.port.read_reply(0x9A, self.cfg.motor_id)
        if len(payload) != 7:
            raise ResponseError(f"状态1长度错误: {len(payload)}")
        temperature = struct.unpack("<b", payload[0:1])[0]
        voltage = self._unpack_i16(payload[1:3]) / 100.0
        current = self._unpack_i16(payload[3:5]) / 100.0
        motor_state = payload[5]
        error_state = payload[6]
        return {
            "temperature_c": temperature,
            "bus_voltage_v": voltage,
            "bus_current_a": current,
            "motor_state": motor_state,
            "error_state": error_state,
        }

    def read_status2(self) -> Dict[str, Any]:
        self.port.send_frame(0x9C, self.cfg.motor_id, b"")
        payload = self.port.read_reply(0x9C, self.cfg.motor_id)
        if len(payload) != 7:
            raise ResponseError(f"状态2长度错误: {len(payload)}")
        temperature = struct.unpack("<b", payload[0:1])[0]
        iq_or_power = self._unpack_i16(payload[1:3])
        speed_dps = self._unpack_i16(payload[3:5])  # 协议里状态2返回单位为 1 dps/LSB
        encoder = self._unpack_u16(payload[5:7])
        return {
            "temperature_c": temperature,
            "iq_or_power_raw": iq_or_power,
            "speed_dps": speed_dps,
            "encoder": encoder,
        }

    def read_multi_turn_angle_deg(self) -> float:
        self.port.send_frame(0x92, self.cfg.motor_id, b"")
        payload = self.port.read_reply(0x92, self.cfg.motor_id)
        if len(payload) != 8:
            raise ResponseError(f"多圈角度长度错误: {len(payload)}")
        angle_raw = self._unpack_i64(payload)
        return self._proto_to_deg(angle_raw)

    def move_to_deg(self, target_deg: float, max_speed_dps: Optional[float] = None) -> None:
        if not (self.cfg.min_deg <= target_deg <= self.cfg.max_deg):
            raise ValueError(
                f"{self.cfg.name} 目标角度超限: {target_deg}, "
                f"允许范围 [{self.cfg.min_deg}, {self.cfg.max_deg}]"
            )
        if max_speed_dps is None:
            max_speed_dps = self.cfg.default_speed_dps

        angle_raw = self._deg_to_proto(target_deg)
        speed_raw = self._speed_to_proto(max_speed_dps)
        payload = self._pack_i64(angle_raw) + self._pack_u32(speed_raw)

        self.port.send_frame(0xA4, self.cfg.motor_id, payload)
        # A4 回复格式与状态2相同
        payload = self.port.read_reply(0xA4, self.cfg.motor_id)
        if len(payload) != 7:
            raise ResponseError(f"A4 回复长度错误: {len(payload)}")

    def center(self, max_speed_dps: Optional[float] = None) -> None:
        self.move_to_deg(0.0, max_speed_dps=max_speed_dps)

    def close(self) -> None:
        self.port.close()


class DualAxisDriver:
    """
    双轴驱动封装：
    - yaw: 左右，电机 ID=1
    - pitch: 俯仰，电机 ID=2
    """
    def __init__(self):
        self.yaw = LkMotor(
            MotorConfig(
                name="yaw",
                motor_id=1,
                dev="/dev/ttySC0",
                txden_pin=27,
                direction_sign=1,   # 先假设 yaw 正方向 = 协议正方向
                default_speed_dps=90.0,
                min_deg=-180.0,
                max_deg=180.0,
            )
        )
        self.pitch = LkMotor(
            MotorConfig(
                name="pitch",
                motor_id=2,
                dev="/dev/ttySC1",
                txden_pin=22,
                direction_sign=-1,  # 与 yaw 相反，先按你的描述给默认值
                default_speed_dps=60.0,
                min_deg=-90.0,
                max_deg=90.0,
            )
        )

    def startup(self) -> None:
        self.yaw.motor_run()
        time.sleep(0.05)
        self.pitch.motor_run()
        time.sleep(0.05)

    def stop_all(self) -> None:
        try:
            self.yaw.motor_stop()
        finally:
            self.pitch.motor_stop()

    def center_all(self) -> None:
        self.yaw.center()
        time.sleep(0.05)
        self.pitch.center()

    def set_pose(self, yaw_deg: float, pitch_deg: float,
                 yaw_speed_dps: Optional[float] = None,
                 pitch_speed_dps: Optional[float] = None) -> None:
        self.yaw.move_to_deg(yaw_deg, yaw_speed_dps)
        time.sleep(0.01)
        self.pitch.move_to_deg(pitch_deg, pitch_speed_dps)

    def get_pose(self) -> Dict[str, float]:
        return {
            "yaw_deg": self.yaw.read_multi_turn_angle_deg(),
            "pitch_deg": self.pitch.read_multi_turn_angle_deg(),
        }

    def get_status(self) -> Dict[str, Dict[str, Any]]:
        return {
            "yaw": {
                "status1": self.yaw.read_status1(),
                "status2": self.yaw.read_status2(),
            },
            "pitch": {
                "status1": self.pitch.read_status1(),
                "status2": self.pitch.read_status2(),
            },
        }

    def close(self) -> None:
        self.yaw.close()
        self.pitch.close()
        if GPIO is not None:
            try:
                GPIO.cleanup()
            except Exception:
                pass


def demo():
    driver = DualAxisDriver()
    try:
        print(f"GPIO backend: {GPIO_BACKEND}")
        print("启动电机...")
        driver.startup()

        print("归中...")
        driver.center_all()
        time.sleep(1.5)

        print("读当前位置:")
        print(driver.get_pose())

        print("移动到 yaw=30°, pitch=15°")
        driver.set_pose(30.0, 15.0, yaw_speed_dps=90.0, pitch_speed_dps=45.0)
        time.sleep(2.0)
        print(driver.get_pose())

        print("移动到 yaw=-30°, pitch=-15°")
        driver.set_pose(-30.0, -15.0, yaw_speed_dps=90.0, pitch_speed_dps=45.0)
        time.sleep(2.0)
        print(driver.get_pose())

        print("停止...")
        driver.stop_all()
    finally:
        driver.close()


if __name__ == "__main__":
    demo()
