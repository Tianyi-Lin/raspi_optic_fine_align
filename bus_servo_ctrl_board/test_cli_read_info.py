import argparse

from transport import SerialTransport
from driver import BusServoBoardDriver


def build_parser():
    parser = argparse.ArgumentParser(
        description="总线舵机控制板测试：读取指定舵机信息"
    )
    parser.add_argument("--port", type=str, default="/dev/ttyAMA1")
    parser.add_argument("--baudrate", type=int, default=9600)
    parser.add_argument("--id", dest="servo_id", type=int, required=True)
    parser.add_argument("--timeout", type=float, default=1.0)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no-board-voltage", action="store_true")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    transport = SerialTransport(
        port=args.port,
        baudrate=args.baudrate,
        timeout=args.timeout,
        debug=args.debug,
    )
    driver = BusServoBoardDriver(transport)

    try:
        print(
            f"[INFO] read: port={args.port}, baudrate={args.baudrate}, "
            f"id={args.servo_id}"
        )

        position = driver.read_one_position(args.servo_id)
        print(f"[INFO] 当前位置: {position}")

        if not args.no_board_voltage:
            try:
                vin = driver.get_battery_voltage_mv()
                print(f"[INFO] 控制板电压: {vin} mV")
            except Exception as e:
                print(f"[WARN] 读取控制板电压失败: {e}")
    finally:
        driver.close()


if __name__ == "__main__":
    main()
