import argparse
import time

from transport import SerialTransport
from driver import BusServoDriver


def build_parser():
    parser = argparse.ArgumentParser(
        description="总线舵机命令行测试：移动到指定位置并回读当前位置"
    )
    parser.add_argument(
        "--port",
        type=str,
        default="/dev/ttyAMA0",
        help="串口设备，例如 /dev/ttyAMA0 或 /dev/ttyUSB0",
    )
    parser.add_argument(
        "--baudrate",
        type=int,
        default=115200,
        help="串口波特率，协议默认 115200",
    )
    parser.add_argument(
        "--id",
        dest="servo_id",
        type=int,
        required=True,
        help="舵机 ID",
    )
    parser.add_argument(
        "--pos",
        type=int,
        required=True,
        help="目标位置，范围 0~1000",
    )
    parser.add_argument(
        "--time",
        dest="time_ms",
        type=int,
        default=500,
        help="运动时间，单位 ms，默认 500",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=0.1,
        help="串口读超时，单位秒，默认 0.1",
    )
    parser.add_argument(
        "--no-readback",
        action="store_true",
        help="只发运动命令，不回读当前位置",
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
        raise ValueError(f"timeout 必须大于 0，当前为 {args.timeout}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    try:
        validate_args(args)
    except ValueError as e:
        parser.error(str(e))

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
            f"[INFO] 发送移动命令: port={args.port}, baudrate={args.baudrate}, "
            f"id={args.servo_id}, pos={args.pos}, time_ms={args.time_ms}"
        )

        driver.move_time_write(args.servo_id, args.pos, args.time_ms)
        print("[INFO] 命令发送成功")

        if not args.no_readback:
            wait_s = args.time_ms / 1000.0 + 0.05
            time.sleep(wait_s)

            current_pos = driver.read_pos(args.servo_id)
            print(f"[INFO] 当前位置: {current_pos}")

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