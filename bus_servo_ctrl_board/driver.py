from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time

from protocol import (
    BoardCmd,
    build_frame,
    parse_frame,
    pack_u16_le,
    unpack_u16_le,
)
from transport import SerialTransport


@dataclass
class ServoPosition:
    servo_id: int
    position: int


class BusServoBoardDriver:
    def __init__(self, transport: SerialTransport):
        self.transport = transport

    def close(self):
        self.transport.close()

    def send_only(self, cmd: int, params=()):
        frame = build_frame(cmd, params)
        self.transport.write(frame)

    def request(
        self,
        cmd: int,
        params=(),
        expect_cmd: Optional[int] = None,
        overall_timeout: float = 0.8,
        retries: int = 2,
    ):
        """
        发一次请求后持续接收。
        不要高频重发。
        """
        last_err = None
        expect = cmd if expect_cmd is None else expect_cmd

        for attempt in range(retries):
            try:
                if hasattr(self.transport, "reset_input_buffer"):
                    self.transport.reset_input_buffer()

                frame = build_frame(cmd, params)
                self.transport.write(frame)

                deadline = time.time() + overall_timeout

                while time.time() < deadline:
                    remain = deadline - time.time()
                    if remain <= 0:
                        break

                    old_timeout = self.transport.timeout
                    self.transport.timeout = min(old_timeout, remain)

                    try:
                        raw = self.transport.read_frame()
                    finally:
                        self.transport.timeout = old_timeout

                    resp = parse_frame(raw)

                    # 混入别的帧就跳过继续等
                    if resp.cmd != expect:
                        continue

                    return resp

                last_err = TimeoutError(
                    f"request timeout: cmd={cmd}, expect_cmd={expect}, "
                    f"attempt={attempt + 1}/{retries}"
                )

            except Exception as e:
                last_err = e

            time.sleep(0.1)

        raise last_err

    # -------------------------
    # 多舵机运动
    # -------------------------
    def move_servos(self, servo_positions: List[Tuple[int, int]], time_ms: int):
        if not servo_positions:
            raise ValueError("servo_positions cannot be empty")
        if not (0 <= time_ms <= 0xFFFF):
            raise ValueError("time_ms out of range")

        params = [len(servo_positions), *pack_u16_le(time_ms)]
        for servo_id, pos in servo_positions:
            if not (0 <= servo_id <= 255):
                raise ValueError(f"invalid servo id: {servo_id}")
            if not (0 <= pos <= 1000):
                raise ValueError(f"invalid position: {pos}")
            params.extend([servo_id, *pack_u16_le(pos)])

        self.send_only(BoardCmd.CMD_SERVO_MOVE, params)

    def move_one(self, servo_id: int, position: int, time_ms: int):
        self.move_servos([(servo_id, position)], time_ms)

    # -------------------------
    # 动作组
    # -------------------------
    def run_action_group(self, group_id: int, times: int):
        if not (0 <= group_id <= 255):
            raise ValueError("group_id out of range")
        if not (0 <= times <= 0xFFFF):
            raise ValueError("times out of range")
        params = [group_id, *pack_u16_le(times)]
        self.send_only(BoardCmd.CMD_ACTION_GROUP_RUN, params)

    def stop_action_group(self):
        self.send_only(BoardCmd.CMD_ACTION_GROUP_STOP)

    def set_action_group_speed(self, group_id: int, percent: int):
        if not (0 <= group_id <= 255):
            raise ValueError("group_id out of range")
        if not (0 <= percent <= 0xFFFF):
            raise ValueError("percent out of range")
        params = [group_id, *pack_u16_le(percent)]
        self.send_only(BoardCmd.CMD_ACTION_GROUP_SPEED, params)

    # -------------------------
    # 读控制板电压
    # -------------------------
    def get_battery_voltage_mv(self) -> int:
        resp = self.request(BoardCmd.CMD_GET_BATTERY_VOLTAGE)
        if len(resp.params) != 2:
            raise ValueError(f"invalid battery voltage response length: {len(resp.params)}")
        return unpack_u16_le(resp.params[0], resp.params[1])

    # -------------------------
    # 多舵机卸力
    # -------------------------
    def unload_servos(self, servo_ids: List[int]):
        if not servo_ids:
            raise ValueError("servo_ids cannot be empty")
        for sid in servo_ids:
            if not (0 <= sid <= 255):
                raise ValueError(f"invalid servo id: {sid}")
        params = [len(servo_ids), *servo_ids]
        self.send_only(BoardCmd.CMD_MULT_SERVO_UNLOAD, params)

    # -------------------------
    # 多舵机读位置
    # -------------------------
    def read_servo_positions(self, servo_ids: List[int]) -> List[ServoPosition]:
        if not servo_ids:
            raise ValueError("servo_ids cannot be empty")
        for sid in servo_ids:
            if not (0 <= sid <= 255):
                raise ValueError(f"invalid servo id: {sid}")

        params = [len(servo_ids), *servo_ids]
        resp = self.request(BoardCmd.CMD_MULT_SERVO_POS_READ, params)

        if len(resp.params) < 1:
            raise ValueError("invalid position response length")

        count = resp.params[0]
        expected_param_len = 1 + count * 3
        if len(resp.params) != expected_param_len:
            raise ValueError(
                f"invalid position response param length: expected {expected_param_len}, got {len(resp.params)}"
            )

        result = []
        for i in range(count):
            base = 1 + i * 3
            sid = resp.params[base]
            pos = unpack_u16_le(resp.params[base + 1], resp.params[base + 2])
            result.append(ServoPosition(servo_id=sid, position=pos))
        return result

    def read_one_position(self, servo_id: int) -> int:
        result = self.read_servo_positions([servo_id])
        if not result:
            raise ValueError("empty position response")
        return result[0].position