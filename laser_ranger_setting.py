# coding: utf-8
import serial
import time
import argparse


def pack_u16_le(v: int):
    return [v & 0xFF, (v >> 8) & 0xFF]


def pack_u24_le(v: int):
    return [v & 0xFF, (v >> 8) & 0xFF, (v >> 16) & 0xFF]


def pack_u32_le(v: int):
    return [
        v & 0xFF,
        (v >> 8) & 0xFF,
        (v >> 16) & 0xFF,
        (v >> 24) & 0xFF,
    ]


def checksum_sum(packet_without_checksum):
    return sum(packet_without_checksum) & 0xFF


def bytes_to_hex(bs):
    return " ".join(f"{b:02X}" for b in bs)


def build_mode_byte(output_mode: str, range_mode: str, interface_mode: str):
    """
    mode:
      bit1    : output mode [0:active, 1:inquire]
      bit2-3  : range mode [00:short, 01:medium, 10:long]
      bit4-5  : interface mode [00:uart, 01:can, 10:io, 11:iic]
    """
    mode = 0

    # bit1
    if output_mode == "active":
        out_bits = 0
    elif output_mode == "inquire":
        out_bits = 1
    else:
        raise ValueError(f"invalid output_mode: {output_mode}")
    mode |= (out_bits << 1)

    # bit2-3
    if range_mode == "short":
        range_bits = 0b00
    elif range_mode == "medium":
        range_bits = 0b01
    elif range_mode == "long":
        range_bits = 0b10
    else:
        raise ValueError(f"invalid range_mode: {range_mode}")
    mode |= (range_bits << 2)

    # bit4-5
    if interface_mode == "uart":
        if_bits = 0b00
    elif interface_mode == "can":
        if_bits = 0b01
    elif interface_mode == "io":
        if_bits = 0b10
    elif interface_mode == "iic":
        if_bits = 0b11
    else:
        raise ValueError(f"invalid interface_mode: {interface_mode}")
    mode |= (if_bits << 4)

    return mode


def build_write_packet(
    module_id=0,
    output_mode="inquire",
    range_mode="medium",
    interface_mode="uart",
    uart_baudrate=115200,
    band_start=0,
    band_width=25000,
):
    """
    按你截图里的 0x54 0x20 设置帧构造写配置包。
    """
    header = 0x54
    func = 0x20
    mix = 0x00  # write
    reserved1 = 0xFF

    # system_time: 终端时间，单位 ms
    system_time_ms = int(time.time() * 1000) & 0xFFFFFFFF

    mode = build_mode_byte(output_mode, range_mode, interface_mode)

    packet = []
    packet.append(header)                     # 0
    packet.append(func)                       # 1
    packet.append(mix)                        # 2
    packet.append(reserved1)                  # 3
    packet.append(module_id & 0xFF)           # 4
    packet.extend(pack_u32_le(system_time_ms))# 5-8
    packet.append(mode)                       # 9
    packet.extend([0xFF, 0xFF])               # 10-11 reserved
    packet.extend(pack_u24_le(uart_baudrate)) # 12-14 uart_baudrate
    packet.append(0xFF)                       # 15 FOV.x
    packet.append(0xFF)                       # 16 FOV.y
    packet.append(0xFF)                       # 17 FOV.x_offset
    packet.append(0xFF)                       # 18 FOV.y_offset
    packet.extend(pack_u16_le(band_start))    # 19-20
    packet.extend(pack_u16_le(band_width))    # 21-22
    packet.extend([0xFF] * 8)                 # 23-30 reserved

    packet.append(checksum_sum(packet))       # 31 checksum
    return bytes(packet)


def build_read_packet(module_id=0):
    """
    构造读取配置命令。
    由于图里 mix bit0=1 表示读，这里按同一帧结构发一个 read 请求。
    未知/无关字段统一填 0xFF。
    """
    header = 0x54
    func = 0x20
    mix = 0x01  # read
    packet = [
        header,
        func,
        mix,
        0xFF,
        module_id & 0xFF,
    ]
    packet.extend([0xFF] * 4)   # system_time
    packet.append(0xFF)         # mode
    packet.extend([0xFF, 0xFF]) # reserved
    packet.extend([0xFF, 0xFF, 0xFF])  # baudrate
    packet.extend([0xFF, 0xFF, 0xFF, 0xFF])  # FOV
    packet.extend([0xFF, 0xFF]) # band_start
    packet.extend([0xFF, 0xFF]) # band_width
    packet.extend([0xFF] * 8)   # reserved
    packet.append(checksum_sum(packet))
    return bytes(packet)


def read_response(ser, timeout=0.5):
    start = time.time()
    buf = bytearray()
    while time.time() - start < timeout:
        n = ser.in_waiting
        if n > 0:
            buf.extend(ser.read(n))
        else:
            time.sleep(0.01)
    return bytes(buf)


def configure_laser_module(
    port="/dev/ttyAMA1", 
    baudrate=115200, 
    module_id=0,
    output_mode="inquire",
    range_mode="medium",
    interface_mode="uart",
    uart_baudrate=115200,
    band_start=0,
    band_width=25000
):
    """
    供外部导入调用的配置函数，用于写入配置并读取确认
    """
    print(f"[LaserConfig] 开始配置激光模块 ({port} @ {baudrate})...")
    try:
        ser = serial.Serial(port, baudrate, timeout=0.2)
        time.sleep(0.1)
        ser.reset_input_buffer()
        
        # 1. 写入配置
        pkt_write = build_write_packet(
            module_id=module_id,
            output_mode=output_mode,
            range_mode=range_mode,
            interface_mode=interface_mode,
            uart_baudrate=uart_baudrate,
            band_start=band_start,
            band_width=band_width,
        )
        print("[LaserConfig] 发送 WRITE 配置帧:", bytes_to_hex(pkt_write))
        ser.write(pkt_write)
        ser.flush()
        
        # 写入后等待 100ms
        time.sleep(0.1)
        # 读取可能的写入确认返回 (清空缓冲区)
        read_response(ser, timeout=0.2)
        
        # 2. 读取配置验证
        pkt_read = build_read_packet(module_id=module_id)
        print("[LaserConfig] 发送 READ 配置帧:", bytes_to_hex(pkt_read))
        ser.write(pkt_read)
        ser.flush()
        
        time.sleep(0.1)
        resp = read_response(ser, timeout=0.5)
        if resp:
            print("[LaserConfig] 读取到的当前配置:", bytes_to_hex(resp))
        else:
            print("[LaserConfig] 警告: 未收到 READ 返回数据")
            
        ser.close()
        print("[LaserConfig] 激光模块配置完成！\n")
        return True
    except Exception as e:
        print(f"[LaserConfig] 配置激光模块时出错: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="TOF Laser Ranger 配置脚本（UART 0x54 0x20）")
    parser.add_argument("--port", default="/dev/ttyAMA1")
    parser.add_argument("--baudrate", type=int, default=115200, help="当前串口波特率")
    parser.add_argument("--id", type=int, default=0, help="模块ID")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # write
    p_write = subparsers.add_parser("write", help="写配置")
    p_write.add_argument("--output-mode", choices=["active", "inquire"], default="inquire")
    p_write.add_argument("--range-mode", choices=["short", "medium", "long"], default="medium")
    p_write.add_argument("--interface-mode", choices=["uart", "can", "io", "iic"], default="uart")
    p_write.add_argument("--uart-baudrate", type=int, default=115200)
    p_write.add_argument("--band-start", type=int, default=0)
    p_write.add_argument("--band-width", type=int, default=25000)

    # read
    p_read = subparsers.add_parser("read", help="读配置")

    args = parser.parse_args()

    ser = serial.Serial(args.port, args.baudrate, timeout=0.2)
    time.sleep(0.1)
    ser.reset_input_buffer()

    try:
        if args.cmd == "write":
            pkt = build_write_packet(
                module_id=args.id,
                output_mode=args.output_mode,
                range_mode=args.range_mode,
                interface_mode=args.interface_mode,
                uart_baudrate=args.uart_baudrate,
                band_start=args.band_start,
                band_width=args.band_width,
            )
            print("[WRITE] 发送配置帧：")
            print(bytes_to_hex(pkt))
            ser.write(pkt)
            ser.flush()

            time.sleep(0.2)
            resp = read_response(ser, timeout=0.5)
            if resp:
                print("[WRITE] 返回数据：")
                print(bytes_to_hex(resp))
            else:
                print("[WRITE] 未收到返回数据（有些设备写配置后不一定立即返回）")

        elif args.cmd == "read":
            pkt = build_read_packet(module_id=args.id)
            print("[READ] 发送读取配置帧：")
            print(bytes_to_hex(pkt))
            ser.write(pkt)
            ser.flush()

            time.sleep(0.2)
            resp = read_response(ser, timeout=0.8)
            if resp:
                print("[READ] 返回数据：")
                print(bytes_to_hex(resp))
            else:
                print("[READ] 未收到返回数据")

    finally:
        ser.close()


if __name__ == "__main__":
    main()