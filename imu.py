# coding: utf-8
from __future__ import annotations
import serial
import threading
import time
from dataclasses import dataclass
from typing import Optional, Dict


def _to_int16(lo: int, hi: int) -> int:
    v = (hi << 8) | lo
    if v >= 32768:
        v -= 65536
    return v


def _to_uint16_from_int16(v: int) -> int:
    v = int(v)
    if v < 0:
        v += 65536
    return v & 0xFFFF


@dataclass
class ImuState:
    acc_x_g: float = 0.0
    acc_y_g: float = 0.0
    acc_z_g: float = 0.0

    gyro_x_dps: float = 0.0
    gyro_y_dps: float = 0.0
    gyro_z_dps: float = 0.0

    roll_deg: float = 0.0
    pitch_deg: float = 0.0
    yaw_deg: float = 0.0

    temperature_c: Optional[float] = None
    last_update: float = 0.0


class IMUReader:
    """
    适配这类 0x55 开头的 JY/Wit 风格 IMU 串口输出。

    支持解析：
    - 0x51: 加速度
    - 0x52: 角速度
    - 0x53: 角度
    """

    FRAME_LEN = 11
    FRAME_HEAD = 0x55
    TYPE_ACC = 0x51
    TYPE_GYRO = 0x52
    TYPE_ANGLE = 0x53

    def __init__(self, port: str, baudrate: int = 9600, timeout: float = 0.1, debug: bool = False):
        self.ser = serial.Serial(port, baudrate=baudrate, timeout=timeout)
        self.debug = debug
        self.buf = bytearray()
        self.state = ImuState()
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._port = port
        self._baudrate = baudrate
        self._timeout = timeout

    # ---------------------------
    # 协议工具
    # ---------------------------
    @staticmethod
    def _checksum_ok(frame: bytes) -> bool:
        return (sum(frame[:10]) & 0xFF) == frame[10]

    def _parse_frame(self, frame: bytes) -> bool:
        """
        解析单帧，成功则更新 state，返回 True
        """
        if len(frame) != 11:
            return False
        if frame[0] != self.FRAME_HEAD:
            return False
        if not self._checksum_ok(frame):
            return False

        ftype = frame[1]
        d = frame[2:10]

        updated = False

        with self._lock:
            if ftype == self.TYPE_ACC:
                ax = _to_int16(d[0], d[1]) / 32768.0 * 16.0
                ay = _to_int16(d[2], d[3]) / 32768.0 * 16.0
                az = _to_int16(d[4], d[5]) / 32768.0 * 16.0
                temp = _to_int16(d[6], d[7]) / 100.0

                self.state.acc_x_g = ax
                self.state.acc_y_g = ay
                self.state.acc_z_g = az
                self.state.temperature_c = temp
                self.state.last_update = time.time()
                updated = True

            elif ftype == self.TYPE_GYRO:
                gx = _to_int16(d[0], d[1]) / 32768.0 * 2000.0
                gy = _to_int16(d[2], d[3]) / 32768.0 * 2000.0
                gz = _to_int16(d[4], d[5]) / 32768.0 * 2000.0

                self.state.gyro_x_dps = gx
                self.state.gyro_y_dps = gy
                self.state.gyro_z_dps = gz
                self.state.last_update = time.time()
                updated = True

            elif ftype == self.TYPE_ANGLE:
                roll = _to_int16(d[0], d[1]) / 32768.0 * 180.0
                pitch = _to_int16(d[2], d[3]) / 32768.0 * 180.0
                yaw = _to_int16(d[4], d[5]) / 32768.0 * 180.0

                self.state.roll_deg = roll
                self.state.pitch_deg = pitch
                self.state.yaw_deg = yaw
                self.state.last_update = time.time()
                updated = True

        return updated

    def _feed(self, data: bytes):
        self.buf.extend(data)

        while len(self.buf) >= self.FRAME_LEN:
            # 对齐帧头
            if self.buf[0] != self.FRAME_HEAD:
                self.buf.pop(0)
                continue

            frame = bytes(self.buf[:self.FRAME_LEN])

            if not self._checksum_ok(frame):
                # 若校验失败，丢一个字节继续找
                self.buf.pop(0)
                continue

            if self.debug:
                print("[IMU FRAME]", frame.hex(" "))

            self._parse_frame(frame)
            del self.buf[:self.FRAME_LEN]

    # ---------------------------
    # 串口寄存器配置
    # ---------------------------
    def write_reg(self, addr: int, value: int):
        """
        写寄存器格式: FF AA ADDR DATAL DATAH
        """
        value = int(value) & 0xFFFF
        pkt = bytes([0xFF, 0xAA, addr & 0xFF, value & 0xFF, (value >> 8) & 0xFF])
        if self.debug:
            print("[IMU TX]", pkt.hex(" "))
        self.ser.write(pkt)
        self.ser.flush()
        time.sleep(0.02)

    def unlock(self):
        # KEY(0x69) = 0xB588
        self.write_reg(0x69, 0xB588)

    def configure_output(self, output_mask: int = 0x000E, rate_code: int = 0x07):
        """
        output_mask:
            bit1 ACC(0x51)
            bit2 GYRO(0x52)
            bit3 ANGLE(0x53)

        推荐默认:
            0x000E = ACC + GYRO + ANGLE

        rate_code:
            RRATE 寄存器值
            常用:
                0x07 -> 20Hz
                0x08 -> 50Hz
        """
        self.unlock()

        # RSW = 输出内容
        self.write_reg(0x02, output_mask)

        # RRATE = 输出速率
        self.write_reg(0x03, rate_code)

        # SAVE = 保存
        self.write_reg(0x00, 0x0000)

    BAUD_CODE_TO_BAUD = {
        0x01: 4800,
        0x02: 9600,
        0x03: 19200,
        0x04: 38400,
        0x05: 57600,
        0x06: 115200,
        0x07: 230400,
        0x08: 460800,
        0x09: 921600,
    }

    BAUD_TO_BAUD_CODE = {v: k for k, v in BAUD_CODE_TO_BAUD.items()}

    RRATE_CODE_TO_HZ = {
        0x06: 10,
        0x07: 20,
        0x08: 50,
        0x09: 100,
        0x0B: 200,
    }

    RRATE_HZ_TO_CODE = {v: k for k, v in RRATE_CODE_TO_HZ.items()}

    def set_output_rate_hz(self, hz: int):
        hz = int(hz)
        if hz not in self.RRATE_HZ_TO_CODE:
            raise ValueError(f"unsupported output rate hz={hz}, supported={sorted(self.RRATE_HZ_TO_CODE.keys())}")
        self.unlock()
        self.write_reg(0x03, int(self.RRATE_HZ_TO_CODE[hz]))
        self.write_reg(0x00, 0x0000)
        time.sleep(0.1)

    def set_algorithm_mode(self, use_6axis: bool = True):
        mode = 0x0001 if bool(use_6axis) else 0x0000
        self.unlock()
        self.write_reg(0x24, mode)
        self.write_reg(0x00, 0x0000)
        time.sleep(0.1)

    def apply_baudrate(self, baudrate: int):
        baudrate = int(baudrate)
        if baudrate not in self.BAUD_TO_BAUD_CODE:
            raise ValueError(f"unsupported baudrate={baudrate}, supported={sorted(self.BAUD_TO_BAUD_CODE.keys())}")
        code = int(self.BAUD_TO_BAUD_CODE[baudrate])
        self.unlock()
        self.write_reg(0x04, code)
        self.write_reg(0x00, 0x0000)
        time.sleep(0.1)
        self.reopen(baudrate=baudrate)

    def set_sensor_offsets(
        self,
        *,
        ax_g: float = 0.0,
        ay_g: float = 0.0,
        az_g: float = 0.0,
        gx_dps: float = 0.0,
        gy_dps: float = 0.0,
        gz_dps: float = 0.0,
        hx: int = 0,
        hy: int = 0,
        hz: int = 0,
    ):
        values = [
            int(round(float(ax_g) * 10000.0)),
            int(round(float(ay_g) * 10000.0)),
            int(round(float(az_g) * 10000.0)),
            int(round(float(gx_dps) * 10000.0)),
            int(round(float(gy_dps) * 10000.0)),
            int(round(float(gz_dps) * 10000.0)),
            int(hx),
            int(hy),
            int(hz),
        ]
        addrs = list(range(0x05, 0x0E))
        self.unlock()
        for addr, v in zip(addrs, values):
            self.write_reg(addr, _to_uint16_from_int16(v))
            time.sleep(0.06)
        self.write_reg(0x00, 0x0000)
        time.sleep(0.1)

    def reopen(self, *, baudrate: Optional[int] = None):
        self.stop()
        try:
            if self.ser and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        if baudrate is not None:
            self._baudrate = int(baudrate)
        self.ser = serial.Serial(self._port, baudrate=self._baudrate, timeout=self._timeout)
        self.buf = bytearray()
        self.start()

    # ---------------------------
    # 读取线程
    # ---------------------------
    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)

    def close(self):
        self.stop()
        if self.ser and self.ser.is_open:
            self.ser.close()

    def _reader_loop(self):
        while self._running:
            try:
                data = self.ser.read(64)
                if data:
                    if self.debug:
                        print("[IMU RX]", data.hex(" "))
                    self._feed(data)
                else:
                    time.sleep(0.005)
            except Exception as e:
                if self.debug:
                    print("[IMU ERROR]", e)
                time.sleep(0.05)

    # ---------------------------
    # 对外接口
    # ---------------------------
    def get_state(self) -> ImuState:
        with self._lock:
            return ImuState(**self.state.__dict__)

    def get_dict(self) -> Dict[str, float]:
        s = self.get_state()
        return {
            "acc_x_g": s.acc_x_g,
            "acc_y_g": s.acc_y_g,
            "acc_z_g": s.acc_z_g,
            "gyro_x_dps": s.gyro_x_dps,
            "gyro_y_dps": s.gyro_y_dps,
            "gyro_z_dps": s.gyro_z_dps,
            "roll_deg": s.roll_deg,
            "pitch_deg": s.pitch_deg,
            "yaw_deg": s.yaw_deg,
            "temperature_c": s.temperature_c if s.temperature_c is not None else 0.0,
            "last_update": s.last_update,
        }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="IMU 串口读取封装示例")
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baudrate", type=int, default=9600)
    ap.add_argument("--configure", action="store_true", help="启动时配置输出内容和速率")
    ap.add_argument("--rate-code", type=lambda x: int(x, 0), default=0x07, help="RRATE寄存器值，如 0x07")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    imu = IMUReader(args.port, baudrate=args.baudrate, debug=args.debug)

    try:
        if args.configure:
            imu.configure_output(output_mask=0x000E, rate_code=args.rate_code)
            time.sleep(0.2)

        imu.start()

        while True:
            s = imu.get_state()
            print(
                f"ACC[g]   x={s.acc_x_g:+7.3f} y={s.acc_y_g:+7.3f} z={s.acc_z_g:+7.3f} | "
                f"GYRO[dps] x={s.gyro_x_dps:+8.3f} y={s.gyro_y_dps:+8.3f} z={s.gyro_z_dps:+8.3f} | "
                f"ANGLE[deg] roll={s.roll_deg:+7.3f} pitch={s.pitch_deg:+7.3f} yaw={s.yaw_deg:+7.3f}"
            )
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[INFO] stopped")
    finally:
        imu.close()
