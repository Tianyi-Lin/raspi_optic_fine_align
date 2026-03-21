# servo 测试脚本

import argparse
import time

from transport import SerialTransport
from driver import BusServoDriver


def build_parser():
    parser = argparse.ArgumentParser(
        description="总线舵机命令行测试：移动到指定位置，并轮询判断是否到位"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyAMA1")
    parser.add_argument("--baudrate", type=int, default=115200)
    parser.add_argument("--id", dest="servo_id", type=int, required=True)
    parser.add_argument("--pos", type=int, required=True, help="目标位置，0~1000")
    parser.add_argument("--time", dest="time_ms", type=int, default=500, help="运动时间 ms")
    parser.add_argument("--timeout", type=float, default=0.5, help="串口读超时，秒")
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="到位容差，默认 ±5",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.1,
        help="轮询位置间隔，默认 0.1 秒",
    )
    parser.add_argument(
        "--max-wait",
        type=float,
        default=None,
        help="最长等待到位时间，秒；默认自动按 time_ms 推算",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="只发送并读一次，不轮询等待到位",
    )
    return parser


def validate_args(args):
    if not (0 <= args.servo_id <= 253):
        raise ValueError(f"舵机 ID 超出范围: {args.servo_id}，应在 0~253")
    if not (0 <= args.pos <= 1000):
        raise ValueError(f"目标位置超出范围: {args.pos}，应在 0~1000")
    if not (0 <= args.time_ms <= 30000):
        raise ValueError(f"运动时间超出范围: {args.time_ms}，应在 0~30000 ms")
    if args.timeout <= 0:
        raise ValueError("timeout 必须大于 0")
    if args.tolerance < 0:
        raise ValueError("tolerance 不能小于 0")
    if args.poll_interval <= 0:
        raise ValueError("poll-interval 必须大于 0")
    if args.max_wait is not None and args.max_wait <= 0:
        raise ValueError("max-wait 必须大于 0")


def read_pos_safe(driver: BusServoDriver, servo_id: int):
    # 底层 driver.read_pos 已经内置了重试机制
    return driver.read_pos(servo_id)


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        parser.error(str(e))

    max_wait = args.max_wait
    if max_wait is None:
        max_wait = max(args.time_ms / 1000.0 + 0.5, 1.0)

    transport = None
    driver = None

    try:
        transport = SerialTransport(
            port=args.port,
            baudrate=args.baudrate,
            timeout=args.timeout,
        )
        driver = BusServoDriver(transport)

        print(
            f"[INFO] move: port={args.port}, baudrate={args.baudrate}, "
            f"id={args.servo_id}, pos={args.pos}, time_ms={args.time_ms}"
        )

        driver.move_time_write(args.servo_id, args.pos, args.time_ms)
        print("[INFO] 命令发送成功")

        # 先等一个基本运动时间
        time.sleep(args.time_ms / 1000.0)

        if args.no_wait:
            current_pos = read_pos_safe(driver, args.servo_id)
            error = current_pos - args.pos
            print(f"[INFO] 当前值: {current_pos}, 误差: {error}")
            return

        start = time.time()
        last_pos = None

        while True:
            current_pos = read_pos_safe(driver, args.servo_id)
            last_pos = current_pos
            error = current_pos - args.pos

            print(f"[INFO] 当前值: {current_pos}, 目标值: {args.pos}, 误差: {error}")

            if abs(error) <= args.tolerance:
                print(f"[OK] 已到位（容差 ±{args.tolerance}）")
                break

            if time.time() - start > max_wait:
                print(
                    f"[WARN] 超时未到位。最后位置: {last_pos}, "
                    f"目标: {args.pos}, 误差: {last_pos - args.pos}"
                )
                break

            time.sleep(args.poll_interval)

    except Exception as e:
        print(f"[ERROR] 执行失败: {e}")
        raise
    finally:
        if driver is not None:
            driver.close()
        elif transport is not None:
            transport.close()


if __name__ == "__main__":
    main()
