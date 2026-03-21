from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence


HEADER = b"\x55\x55"
BROADCAST_ID = 0xFE


class ServoCmd:
    SERVO_MOVE_TIME_WRITE = 1
    SERVO_MOVE_TIME_READ = 2
    SERVO_MOVE_TIME_WAIT_WRITE = 7
    SERVO_MOVE_TIME_WAIT_READ = 8
    SERVO_MOVE_START = 11
    SERVO_MOVE_STOP = 12
    SERVO_ID_WRITE = 13
    SERVO_ID_READ = 14
    SERVO_ANGLE_OFFSET_ADJUST = 17
    SERVO_ANGLE_OFFSET_WRITE = 18
    SERVO_ANGLE_OFFSET_READ = 19
    SERVO_ANGLE_LIMIT_WRITE = 20
    SERVO_ANGLE_LIMIT_READ = 21
    SERVO_VIN_LIMIT_WRITE = 22
    SERVO_VIN_LIMIT_READ = 23
    SERVO_TEMP_MAX_LIMIT_WRITE = 24
    SERVO_TEMP_MAX_LIMIT_READ = 25
    SERVO_TEMP_READ = 26
    SERVO_VIN_READ = 27
    SERVO_POS_READ = 28
    SERVO_OR_MOTOR_MODE_WRITE = 29
    SERVO_OR_MOTOR_MODE_READ = 30
    SERVO_LOAD_OR_UNLOAD_WRITE = 31
    SERVO_LOAD_OR_UNLOAD_READ = 32
    SERVO_LED_CTRL_WRITE = 33
    SERVO_LED_CTRL_READ = 34
    SERVO_LED_ERROR_WRITE = 35
    SERVO_LED_ERROR_READ = 36
    SERVO_DIS_READ = 48


@dataclass
class ServoFrame:
    servo_id: int
    length: int
    cmd: int
    params: bytes
    checksum: int


def checksum(servo_id: int, length: int, cmd: int, params: Sequence[int]) -> int:
    s = (servo_id + length + cmd + sum(params)) & 0xFF
    return (~s) & 0xFF


def pack_u16_le(value: int) -> List[int]:
    if not (0 <= value <= 0xFFFF):
        raise ValueError(f"u16 out of range: {value}")
    return [value & 0xFF, (value >> 8) & 0xFF]


def unpack_u16_le(lo: int, hi: int) -> int:
    return (hi << 8) | lo


def pack_i16_le(value: int) -> List[int]:
    if not (-32768 <= value <= 32767):
        raise ValueError(f"i16 out of range: {value}")
    value &= 0xFFFF
    return [value & 0xFF, (value >> 8) & 0xFF]


def unpack_i16_le(lo: int, hi: int) -> int:
    value = (hi << 8) | lo
    if value & 0x8000:
        value -= 0x10000
    return value


def pack_i8(value: int) -> int:
    if not (-128 <= value <= 127):
        raise ValueError(f"i8 out of range: {value}")
    return value & 0xFF


def unpack_i8(value: int) -> int:
    return value - 256 if value & 0x80 else value


def build_frame(servo_id: int, cmd: int, params: Sequence[int] = ()) -> bytes:
    if not (0 <= servo_id <= 0xFE):
        raise ValueError(f"invalid servo id: {servo_id}")
    params = list(params)
    length = len(params) + 3
    chk = checksum(servo_id, length, cmd, params)
    return HEADER + bytes([servo_id, length, cmd, *params, chk])


def parse_frame(data: bytes) -> ServoFrame:
    if len(data) < 6:
        raise ValueError("frame too short")
    if data[:2] != HEADER:
        raise ValueError("invalid header")

    servo_id = data[2]
    length = data[3]
    expected_len = length + 3
    if len(data) != expected_len:
        raise ValueError(f"invalid frame length: expected {expected_len}, got {len(data)}")

    cmd = data[4]
    params = data[5:-1]
    chk = data[-1]
    calc = checksum(servo_id, length, cmd, params)
    if chk != calc:
        raise ValueError(f"checksum mismatch: got {chk:#04x}, expected {calc:#04x}")

    return ServoFrame(
        servo_id=servo_id,
        length=length,
        cmd=cmd,
        params=params,
        checksum=chk,
    )