import argparse
import time

from transport import SerialTransport
from driver import BusServoBoardDriver


def build_parser():
    parser = argparse.ArgumentParser(
        description="总线舵机控制板测试：移动到指定位置并读回位置"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyAMA0")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--id", dest="servo_id", type=int, required=True)
    parser.add_argument("--pos", type=int, required=True, help="目标位置，0~1000")
    parser.add_argument("--time", dest="time_ms", type=int, default=500, help="运动时间 ms")
    parser.add_argument("--timeout", type=float, default=0.2)
    parser.add_argument("--no-readback", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    transport = SerialTransport(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
    )
    driver = BusServoBoardDriver(transport)

    try:
        print(
            f"[INFO] move: port={args.port}, baudrate={args.baudrate}, "
            f"id={args.servo_id}, pos={args.pos}, time_ms={args.time_ms}"
        )

        driver.move_one(args.servo_id, args.pos, args.time_ms)
        print("[INFO] 命令发送成功")

        if not args.no_readback:
            time.sleep(args.time_ms / 1000.0 + 0.1)
            current_pos = driver.read_one_position(args.servo_id)
            print(f"[INFO] 当前位置: {current_pos}")

        try:
            vin = driver.get_battery_voltage_mv()
            print(f"[INFO] 控制板电压: {vin} mV")
        except Exception as e:
            print(f"[WARN] 读取控制板电压失败: {e}")

    finally:
        driver.close()


if __name__ == "__main__":
    main()