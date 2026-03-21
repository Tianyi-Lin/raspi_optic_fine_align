from __future__ import annotations
from dataclasses import dataclass
from typing import List, Sequence


HEADER = b"\x55\x55"


class BoardCmd:
    CMD_SERVO_MOVE = 3
    CMD_ACTION_GROUP_RUN = 6
    CMD_ACTION_GROUP_STOP = 7
    CMD_ACTION_GROUP_COMPLETE = 8
    CMD_ACTION_GROUP_SPEED = 11
    CMD_GET_BATTERY_VOLTAGE = 15
    CMD_MULT_SERVO_UNLOAD = 20
    CMD_MULT_SERVO_POS_READ = 21


@dataclass
class BoardFrame:
    length: int
    cmd: int
    params: bytes


def pack_u16_le(value: int) -> List[int]:
    if not (0 <= value <= 0xFFFF):
        raise ValueError(f"u16 out of range: {value}")
    return [value & 0xFF, (value >> 8) & 0xFF]


def unpack_u16_le(lo: int, hi: int) -> int:
    return (hi << 8) | lo


def build_frame(cmd: int, params: Sequence[int] = ()) -> bytes:
    params = list(params)
    for p in params:
        if not (0 <= p <= 0xFF):
            raise ValueError(f"param byte out of range: {p}")
    length = len(params) + 2
    return HEADER + bytes([length, cmd, *params])


def parse_frame(data: bytes) -> BoardFrame:
    if len(data) < 4:
        raise ValueError("frame too short")
    if data[:2] != HEADER:
        raise ValueError("invalid header")

    length = data[2]
    if length < 2:
        raise ValueError(f"invalid length: {length}")

    expected_len = length + 2
    if len(data) != expected_len:
        raise ValueError(f"invalid frame length: expected {expected_len}, got {len(data)}")

    cmd = data[3]
    params = data[4:]
    return BoardFrame(length=length, cmd=cmd, params=params)