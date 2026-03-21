from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

from protocol import (
    ServoCmd,
    BROADCAST_ID,
    build_frame,
    parse_frame,
    pack_u16_le,
    unpack_u16_le,
    pack_i16_le,
    unpack_i16_le,
    pack_i8,
    unpack_i8,
)
from transport import SerialTransport


@dataclass
class MoveTime:
    position: int
    time_ms: int


class BusServoDriver:
    def __init__(self, transport: SerialTransport):
        self.transport = transport

    def close(self):
        self.transport.close()

    def send_only(self, servo_id: int, cmd: int, params=()):
        frame = build_frame(servo_id, cmd, params)
        self.transport.write(frame)

    def request(self, servo_id: int, cmd: int, params=(), *, expect_cmd: Optional[int] = None):
        frame = build_frame(servo_id, cmd, params)

        # 关键：发读命令前清理旧缓存 
        if hasattr(self.transport, "ser"): 
            self.transport.ser.reset_input_buffer() 

        self.transport.write(frame)
        raw = self.transport.read_frame()
        resp = parse_frame(raw)

        if resp.servo_id != servo_id and servo_id != BROADCAST_ID:
            raise ValueError(f"unexpected servo id: {resp.servo_id}, expect {servo_id}")

        if expect_cmd is None:
            expect_cmd = cmd
        if resp.cmd != expect_cmd:
            raise ValueError(f"unexpected cmd: {resp.cmd}, expect {expect_cmd}")
        return resp

    # -------------------------
    # 运动类
    # -------------------------
    def move_time_write(self, servo_id: int, position: int, time_ms: int):
        self._check_position(position)
        self._check_time_ms(time_ms)
        params = [*pack_u16_le(position), *pack_u16_le(time_ms)]
        self.send_only(servo_id, ServoCmd.SERVO_MOVE_TIME_WRITE, params)

    def move_time_wait_write(self, servo_id: int, position: int, time_ms: int):
        self._check_position(position)
        self._check_time_ms(time_ms)
        params = [*pack_u16_le(position), *pack_u16_le(time_ms)]
        self.send_only(servo_id, ServoCmd.SERVO_MOVE_TIME_WAIT_WRITE, params)

    def move_start(self, servo_id: int):
        self.send_only(servo_id, ServoCmd.SERVO_MOVE_START)

    def move_stop(self, servo_id: int):
        self.send_only(servo_id, ServoCmd.SERVO_MOVE_STOP)

    def read_move_time(self, servo_id: int) -> MoveTime:
        resp = self.request(servo_id, ServoCmd.SERVO_MOVE_TIME_READ)
        if len(resp.params) != 4:
            raise ValueError("invalid move_time response length")
        pos = unpack_u16_le(resp.params[0], resp.params[1])
        t = unpack_u16_le(resp.params[2], resp.params[3])
        return MoveTime(position=pos, time_ms=t)

    def read_move_time_wait(self, servo_id: int) -> MoveTime:
        resp = self.request(servo_id, ServoCmd.SERVO_MOVE_TIME_WAIT_READ)
        if len(resp.params) != 4:
            raise ValueError("invalid move_time_wait response length")
        pos = unpack_u16_le(resp.params[0], resp.params[1])
        t = unpack_u16_le(resp.params[2], resp.params[3])
        return MoveTime(position=pos, time_ms=t)

    # -------------------------
    # ID
    # -------------------------
    def write_id(self, servo_id: int, new_id: int):
        self._check_id(new_id)
        self.send_only(servo_id, ServoCmd.SERVO_ID_WRITE, [new_id])

    def read_id(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_ID_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid id response length")
        return resp.params[0]

    def discover_single_servo_id(self) -> int:
        """
        只适用于总线上仅有一个舵机时，使用广播 ID 读取。
        """
        resp = self.request(BROADCAST_ID, ServoCmd.SERVO_ID_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid id response length")
        return resp.params[0]

    # -------------------------
    # 偏差
    # -------------------------
    def angle_offset_adjust(self, servo_id: int, offset: int):
        if not (-125 <= offset <= 125):
            raise ValueError("offset must be in [-125, 125]")
        self.send_only(servo_id, ServoCmd.SERVO_ANGLE_OFFSET_ADJUST, [pack_i8(offset)])

    def angle_offset_write(self, servo_id: int):
        self.send_only(servo_id, ServoCmd.SERVO_ANGLE_OFFSET_WRITE)

    def read_angle_offset(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_ANGLE_OFFSET_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid angle offset response length")
        return unpack_i8(resp.params[0])

    # -------------------------
    # 限位
    # -------------------------
    def write_angle_limit(self, servo_id: int, min_pos: int, max_pos: int):
        self._check_position(min_pos)
        self._check_position(max_pos)
        if min_pos >= max_pos:
            raise ValueError("min_pos must be < max_pos")
        params = [*pack_u16_le(min_pos), *pack_u16_le(max_pos)]
        self.send_only(servo_id, ServoCmd.SERVO_ANGLE_LIMIT_WRITE, params)

    def read_angle_limit(self, servo_id: int):
        resp = self.request(servo_id, ServoCmd.SERVO_ANGLE_LIMIT_READ)
        if len(resp.params) != 4:
            raise ValueError("invalid angle limit response length")
        min_pos = unpack_u16_le(resp.params[0], resp.params[1])
        max_pos = unpack_u16_le(resp.params[2], resp.params[3])
        return min_pos, max_pos

    def write_vin_limit(self, servo_id: int, min_mv: int, max_mv: int):
        if not (4500 <= min_mv <= 14000 and 4500 <= max_mv <= 14000):
            raise ValueError("vin limit out of range")
        if min_mv >= max_mv:
            raise ValueError("min_mv must be < max_mv")
        params = [*pack_u16_le(min_mv), *pack_u16_le(max_mv)]
        self.send_only(servo_id, ServoCmd.SERVO_VIN_LIMIT_WRITE, params)

    def read_vin_limit(self, servo_id: int):
        resp = self.request(servo_id, ServoCmd.SERVO_VIN_LIMIT_READ)
        if len(resp.params) != 4:
            raise ValueError("invalid vin limit response length")
        min_mv = unpack_u16_le(resp.params[0], resp.params[1])
        max_mv = unpack_u16_le(resp.params[2], resp.params[3])
        return min_mv, max_mv

    # -------------------------
    # 温度 / 电压 / 位置
    # -------------------------
    def write_temp_max_limit(self, servo_id: int, temp_c: int):
        if not (50 <= temp_c <= 100):
            raise ValueError("temp limit must be in [50, 100]")
        self.send_only(servo_id, ServoCmd.SERVO_TEMP_MAX_LIMIT_WRITE, [temp_c])

    def read_temp_max_limit(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_TEMP_MAX_LIMIT_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid temp max limit response length")
        return resp.params[0]

    def read_temp(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_TEMP_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid temp response length")
        return resp.params[0]

    def read_vin(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_VIN_READ)
        if len(resp.params) != 2:
            raise ValueError("invalid vin response length")
        return unpack_u16_le(resp.params[0], resp.params[1])

    def read_pos(self, servo_id: int, retries: int = 3, retry_gap: float = 0.03) -> int:
        import time
        last_err = None

        for i in range(retries):
            try:
                resp = self.request(servo_id, ServoCmd.SERVO_POS_READ)
                if len(resp.params) != 2:
                    raise ValueError("invalid pos response length")
                return unpack_i16_le(resp.params[0], resp.params[1])
            except Exception as e:
                last_err = e
                time.sleep(retry_gap)

        raise last_err

    # -------------------------
    # 模式
    # -------------------------
    def write_servo_mode(self, servo_id: int):
        # mode=0, turn_mode=0, speed=0
        params = [0, 0, *pack_i16_le(0)]
        self.send_only(servo_id, ServoCmd.SERVO_OR_MOTOR_MODE_WRITE, params)

    def write_motor_mode(self, servo_id: int, speed: int, turn_mode: int = 0):
        if turn_mode not in (0, 1):
            raise ValueError("turn_mode must be 0 or 1")
        if turn_mode == 0:
            if not (-1000 <= speed <= 1000):
                raise ValueError("duty mode speed must be in [-1000, 1000]")
        else:
            if not (-50 <= speed <= 50):
                raise ValueError("rpm mode speed must be in [-50, 50]")

        params = [1, turn_mode, *pack_i16_le(speed)]
        self.send_only(servo_id, ServoCmd.SERVO_OR_MOTOR_MODE_WRITE, params)

    def read_mode(self, servo_id: int) -> Dict[str, Any]:
        resp = self.request(servo_id, ServoCmd.SERVO_OR_MOTOR_MODE_READ)
        if len(resp.params) != 4:
            raise ValueError("invalid mode response length")
        mode = resp.params[0]
        turn_mode = resp.params[1]
        speed = unpack_i16_le(resp.params[2], resp.params[3])
        return {
            "mode": mode,          # 0 servo, 1 motor
            "turn_mode": turn_mode,
            "speed": speed,
        }

    # -------------------------
    # 上电/卸载
    # -------------------------
    def set_load(self, servo_id: int, enable: bool):
        self.send_only(
            servo_id,
            ServoCmd.SERVO_LOAD_OR_UNLOAD_WRITE,
            [1 if enable else 0]
        )

    def read_load_state(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_LOAD_OR_UNLOAD_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid load state response length")
        return resp.params[0]

    # -------------------------
    # LED
    # -------------------------
    def set_led_ctrl(self, servo_id: int, off: bool):
        # 协议：0常亮, 1常灭
        self.send_only(servo_id, ServoCmd.SERVO_LED_CTRL_WRITE, [1 if off else 0])

    def read_led_ctrl(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_LED_CTRL_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid led ctrl response length")
        return resp.params[0]

    def set_led_error(self, servo_id: int, mask: int):
        if not (0 <= mask <= 7):
            raise ValueError("led error mask must be in [0, 7]")
        self.send_only(servo_id, ServoCmd.SERVO_LED_ERROR_WRITE, [mask])

    def read_led_error(self, servo_id: int) -> int:
        resp = self.request(servo_id, ServoCmd.SERVO_LED_ERROR_READ)
        if len(resp.params) != 1:
            raise ValueError("invalid led error response length")
        return resp.params[0]

    # -------------------------
    # 工具函数
    # -------------------------
    @staticmethod
    def _check_id(servo_id: int):
        if not (0 <= servo_id <= 253):
            raise ValueError("servo id must be in [0, 253]")

    @staticmethod
    def _check_position(position: int):
        if not (0 <= position <= 1000):
            raise ValueError("position must be in [0, 1000]")

    @staticmethod
    def _check_time_ms(time_ms: int):
        if not (0 <= time_ms <= 30000):
            raise ValueError("time_ms must be in [0, 30000]")