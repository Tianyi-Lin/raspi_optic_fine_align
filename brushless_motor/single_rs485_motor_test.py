#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import time

from dual_rs485_motor_driver_v1 import MotorConfig, LkMotor, GPIO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="single_rs485_motor_test", description="单电机 RS485 测试脚本")
    parser.add_argument("--id", type=int, required=True, help="电机ID，例如 1 或 2")
    parser.add_argument("--dev", type=str, required=True, help="串口设备，例如 /dev/ttySC0")
    parser.add_argument("--txden", type=int, required=True, help="485方向控制GPIO，BCM编号")
    parser.add_argument("--baudrate", type=int, default=1000000, help="串口波特率")
    parser.add_argument("--timeout", type=float, default=0.05, help="串口超时秒")
    parser.add_argument("--dir-sign", type=int, choices=[-1, 1], default=1, help="角度方向映射")
    parser.add_argument("--min-deg", type=float, default=-180.0, help="最小角度限制")
    parser.add_argument("--max-deg", type=float, default=180.0, help="最大角度限制")
    parser.add_argument("--default-speed", type=float, default=90.0, help="默认速度 dps")
    parser.add_argument("--run", action="store_true", help="先发送电机运行命令")
    parser.add_argument("--stop", action="store_true", help="发送电机停止命令")
    parser.add_argument("--off", action="store_true", help="发送电机关闭命令")
    parser.add_argument("--angle", type=float, default=None, help="目标角度，单位度")
    parser.add_argument("--speed", type=float, default=None, help="目标角速度上限 dps")
    parser.add_argument("--center", action="store_true", help="移动到0度")
    parser.add_argument("--read-angle", action="store_true", help="读取多圈角度")
    parser.add_argument("--status", action="store_true", help="读取状态1和状态2")
    parser.add_argument("--wait", type=float, default=0.2, help="动作后等待秒数")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    cfg = MotorConfig(
        name=f"motor_{args.id}",
        motor_id=args.id,
        dev=args.dev,
        txden_pin=args.txden,
        direction_sign=args.dir_sign,
        baudrate=args.baudrate,
        timeout=args.timeout,
        default_speed_dps=args.default_speed,
        min_deg=args.min_deg,
        max_deg=args.max_deg,
    )
    motor = LkMotor(cfg)
    try:
        if args.run:
            motor.motor_run()
            print("RUN: ok")
        if args.stop:
            motor.motor_stop()
            print("STOP: ok")
        if args.off:
            motor.motor_off()
            print("OFF: ok")
        if args.center:
            motor.center(max_speed_dps=args.speed)
            print(f"CENTER: speed={args.speed if args.speed is not None else cfg.default_speed_dps} dps")
            time.sleep(max(0.0, args.wait))
        if args.angle is not None:
            motor.move_to_deg(args.angle, max_speed_dps=args.speed)
            print(
                f"MOVE: id={args.id} angle={args.angle:.3f} deg "
                f"speed={args.speed if args.speed is not None else cfg.default_speed_dps} dps"
            )
            time.sleep(max(0.0, args.wait))
        if args.read_angle:
            angle = motor.read_multi_turn_angle_deg()
            print(f"ANGLE: {angle:.3f} deg")
        if args.status:
            s1 = motor.read_status1()
            s2 = motor.read_status2()
            print(f"STATUS1: {s1}")
            print(f"STATUS2: {s2}")
    finally:
        motor.close()
        if GPIO is not None:
            try:
                GPIO.cleanup()
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)
