import argparse
import time

from motor_align import Motor


def generate_positions(start, end, step):
    if step <= 0:
        raise ValueError("step 必须大于 0")
    forward = []
    value = start
    while value <= end:
        forward.append(round(value, 4))
        value += step
    if forward[-1] != end:
        forward.append(end)
    backward = list(reversed(forward[:-1]))
    return forward + backward


def run_test(motor, positions, hold_sec, cycles, read_reply):
    cycle = 0
    while cycles <= 0 or cycle < cycles:
        cycle += 1
        for angle in positions:
            print(f"cycle={cycle}, send_angle={angle}")
            motor.send_command(angle)
            time.sleep(hold_sec)
            if read_reply:
                motor.read_reply()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", default="/dev/ttyAMA0")
    parser.add_argument("--motor-id", type=int, default=1)
    parser.add_argument("--max-speed", type=int, default=3000)
    parser.add_argument("--start", type=float, default=-10.0)
    parser.add_argument("--end", type=float, default=10.0)
    parser.add_argument("--step", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=0.2)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--read-reply", action="store_true")
    args = parser.parse_args()

    if args.start >= args.end:
        raise ValueError("start 必须小于 end")
    if args.hold <= 0:
        raise ValueError("hold 必须大于 0")

    positions = generate_positions(args.start, args.end, args.step)
    motor = Motor(serial_port=args.port, max_speed=args.max_speed, motor_id=args.motor_id)

    try:
        run_test(
            motor=motor,
            positions=positions,
            hold_sec=args.hold,
            cycles=args.cycles,
            read_reply=args.read_reply,
        )
    except KeyboardInterrupt:
        print("stopped by user")
    finally:
        motor.close_motor()
        print("serial closed")


if __name__ == "__main__":
    main()
