# coding: UTF-8
import serial
import time
import threading
import tkinter as tk
from tkinter import ttk
import collections
from datetime import datetime

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class LaserRangerQueryMonitor:
    """
    NLink_TOFSense UART 查询模式监控版
    - 定时发送 Read_Frame0 查询命令
    - 接收并解析返回的 Frame0 数据
    """

    def __init__(self, port="/dev/ttyAMA1", baudrate=115200, module_id=0, query_interval=0.1, history_len=500):
        self.port = port
        self.baudrate = baudrate
        self.module_id = module_id
        self.query_interval = query_interval
        self.serial = None
        self.running = False
        self.read_thread = None

        # 协议
        self.FRAME_LEN = 16
        self.REQ_LEN = 8
        self.HEADER = 0x57
        self.FRAME0_FUNC = 0x00
        self.READ_FUNC = 0x10

        # 当前信息
        self.frame_header = 0
        self.function_mark = 0
        self.reserved0 = 0
        self.resp_module_id = 0
        self.system_time_ms = 0
        self.distance_mm = -1.0
        self.distance_m = -1.0
        self.dis_status = 0
        self.valid = 0
        self.signal_strength = 0
        self.range_precision_cm = 0
        self.checksum_ok = False
        self.last_checksum = 0
        self.last_update_str = "-"
        self.last_raw_packet = ""
        self.last_request_packet = ""

        # 统计
        self.tx_queries = 0
        self.rx_frames = 0
        self.bad_checksum_frames = 0
        self.timeout_frames = 0

        # 曲线数据
        self.start_time = time.time()
        self.time_data = collections.deque(maxlen=history_len)
        self.distance_m_data = collections.deque(maxlen=history_len)
        self.signal_data = collections.deque(maxlen=history_len)
        self.status_data = collections.deque(maxlen=history_len)
        self.valid_data = collections.deque(maxlen=history_len)
        self.precision_data = collections.deque(maxlen=history_len)

        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.2)
            self.serial.reset_input_buffer()
            print(f"[LaserRangerQueryMonitor] 已连接串口 {self.port}")
        except Exception as e:
            print(f"[LaserRangerQueryMonitor] 串口连接失败: {e}")

    def packet_to_hex(self, packet):
        return " ".join(f"{b:02X}" for b in packet)

    def verify_checksum(self, data):
        return (sum(data[:-1]) % 256) == data[-1]

    def status_to_valid(self, dis_status):
        # 先按你前面的监控逻辑保留
        return 1 if dis_status == 1 else 0

    def build_query_packet(self, module_id=0):
        """
        依据你给的协议图：
        57 10 FF FF 00 FF FF 63
        若 module_id != 0，则替换第 5 个字节
        checksum = sum(前7字节) % 256
        """
        packet = [0x57, 0x10, 0xFF, 0xFF, module_id & 0xFF, 0xFF, 0xFF]
        packet.append(sum(packet) % 256)
        return bytes(packet)

    def start(self):
        if self.serial is None or not self.serial.is_open:
            print("[LaserRangerQueryMonitor] 串口未打开，无法启动")
            return
        self.running = True
        self.read_thread = threading.Thread(target=self._query_loop, daemon=True)
        self.read_thread.start()

    def stop(self):
        self.running = False
        if self.read_thread is not None:
            self.read_thread.join(timeout=1.0)
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            print("[LaserRangerQueryMonitor] 串口已关闭")

    def query_once(self):
        """单次发起查询并阻塞等待返回（用于按需测量）"""
        if self.serial is None or not self.serial.is_open:
            return False
            
        try:
            # 发送查询命令
            req = self.build_query_packet(self.module_id)
            self.last_request_packet = self.packet_to_hex(req)
            self.serial.reset_input_buffer()
            self.serial.write(req)
            self.serial.flush()
            self.tx_queries += 1

            # 读取返回帧
            packet = self._read_one_frame(timeout=self.serial.timeout)

            if packet is None:
                self.timeout_frames += 1
                return False

            self.rx_frames += 1
            self.last_raw_packet = self.packet_to_hex(packet)
            self.last_checksum = packet[-1]

            checksum_ok = self.verify_checksum(packet)
            self.checksum_ok = checksum_ok
            if not checksum_ok:
                self.bad_checksum_frames += 1
                return False

            # 解析 Frame0
            self.frame_header = packet[0]
            self.function_mark = packet[1]
            self.reserved0 = packet[2]
            self.resp_module_id = packet[3]

            self.system_time_ms = (
                packet[4]
                | (packet[5] << 8)
                | (packet[6] << 16)
                | (packet[7] << 24)
            )

            self.distance_mm = float(
                packet[8]
                | (packet[9] << 8)
                | (packet[10] << 16)
            )
            self.distance_m = self.distance_mm / 1000.0

            self.dis_status = packet[11]
            self.valid = self.status_to_valid(self.dis_status)

            self.signal_strength = packet[12] | (packet[13] << 8)
            self.range_precision_cm = packet[14]

            self.last_update_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            t = time.time() - self.start_time
            self.time_data.append(t)
            self.distance_m_data.append(self.distance_m)
            self.signal_data.append(self.signal_strength)
            self.status_data.append(self.dis_status)
            self.valid_data.append(self.valid)
            self.precision_data.append(self.range_precision_cm)
            
            return True

        except Exception as e:
            print(f"[LaserRangerQueryMonitor] 单次查询错误: {e}")
            return False

    def _query_loop(self):
        while self.running and self.serial and self.serial.is_open:
            try:
                # 发送查询命令
                req = self.build_query_packet(self.module_id)
                self.last_request_packet = self.packet_to_hex(req)
                self.serial.reset_input_buffer()
                self.serial.write(req)
                self.serial.flush()
                self.tx_queries += 1

                # 读取返回帧
                packet = self._read_one_frame(timeout=self.serial.timeout)

                if packet is None:
                    self.timeout_frames += 1
                    time.sleep(self.query_interval)
                    continue

                self.rx_frames += 1
                self.last_raw_packet = self.packet_to_hex(packet)
                self.last_checksum = packet[-1]

                checksum_ok = self.verify_checksum(packet)
                self.checksum_ok = checksum_ok
                if not checksum_ok:
                    self.bad_checksum_frames += 1
                    time.sleep(self.query_interval)
                    continue

                # 解析 Frame0
                self.frame_header = packet[0]
                self.function_mark = packet[1]
                self.reserved0 = packet[2]
                self.resp_module_id = packet[3]

                self.system_time_ms = (
                    packet[4]
                    | (packet[5] << 8)
                    | (packet[6] << 16)
                    | (packet[7] << 24)
                )

                self.distance_mm = float(
                    packet[8]
                    | (packet[9] << 8)
                    | (packet[10] << 16)
                )
                self.distance_m = self.distance_mm / 1000.0

                self.dis_status = packet[11]
                self.valid = self.status_to_valid(self.dis_status)

                self.signal_strength = packet[12] | (packet[13] << 8)
                self.range_precision_cm = packet[14]

                self.last_update_str = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                t = time.time() - self.start_time
                self.time_data.append(t)
                self.distance_m_data.append(self.distance_m)
                self.signal_data.append(self.signal_strength)
                self.status_data.append(self.dis_status)
                self.valid_data.append(self.valid)
                self.precision_data.append(self.range_precision_cm)

                time.sleep(self.query_interval)

            except Exception as e:
                print(f"[LaserRangerQueryMonitor] 查询错误: {e}")
                time.sleep(0.1)

    def _read_one_frame(self, timeout=0.2):
        """
        读一个完整 16 字节 Frame0 返回帧
        """
        start = time.time()
        buffer = []

        while time.time() - start < timeout + 0.3:
            waiting = self.serial.in_waiting
            if waiting > 0:
                raw = self.serial.read(waiting)
                buffer.extend(list(raw))

                while len(buffer) >= self.FRAME_LEN:
                    header_idx = -1
                    for i in range(len(buffer)):
                        if buffer[i] == self.HEADER:
                            header_idx = i
                            break

                    if header_idx == -1:
                        buffer.clear()
                        break

                    if len(buffer) - header_idx < self.FRAME_LEN:
                        buffer = buffer[header_idx:]
                        break

                    packet = buffer[header_idx: header_idx + self.FRAME_LEN]
                    buffer = buffer[header_idx + self.FRAME_LEN:]

                    # 只收查询返回的 Frame0
                    if packet[0] == 0x57 and packet[1] == self.FRAME0_FUNC:
                        return packet
            else:
                time.sleep(0.005)

        return None


class MonitorGUI:
    def __init__(self, root, monitor):
        self.root = root
        self.monitor = monitor
        self.root.title("TOF Laser Ranger Query Monitor / UART ttyAMA1")
        self.root.geometry("1320x980")

        self._build_top_info_panel()
        self._build_plot_panel()
        self.update_gui()

    def _build_top_info_panel(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        title = ttk.Label(
            top_frame,
            text="TOF 激光测距监控面板（UART 查询模式 /dev/ttyAMA1）",
            font=("Arial", 16, "bold")
        )
        title.pack(anchor="w", pady=(0, 8))

        info_frame = ttk.Frame(top_frame)
        info_frame.pack(fill=tk.X)

        self.info_vars = {}

        fields = [
            ("串口", "port"),
            ("波特率", "baudrate"),
            ("查询模块ID", "query_module_id"),
            ("查询周期(s)", "query_interval"),
            ("最近查询帧", "last_request_packet"),
            ("帧头", "frame_header"),
            ("功能码", "function_mark"),
            ("保留字节", "reserved0"),
            ("返回模块ID", "resp_module_id"),
            ("System Time (ms)", "system_time_ms"),
            ("距离 (mm)", "distance_mm"),
            ("距离 (m)", "distance_m"),
            ("状态 Status", "dis_status"),
            ("是否有效 Valid", "valid"),
            ("信号强度 Signal", "signal_strength"),
            ("精度 Precision (cm)", "range_precision_cm"),
            ("校验通过", "checksum_ok"),
            ("发送查询数", "tx_queries"),
            ("接收帧数", "rx_frames"),
            ("超时次数", "timeout_frames"),
            ("校验失败帧数", "bad_checksum_frames"),
            ("最近更新时间", "last_update_str"),
            ("最近原始帧", "last_raw_packet"),
        ]

        for i, (label_text, key) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2

            ttk.Label(info_frame, text=label_text + "：", font=("Arial", 11, "bold")).grid(
                row=row, column=col, sticky="nw", padx=5, pady=3
            )

            var = tk.StringVar(value="-")
            self.info_vars[key] = var

            ttk.Label(
                info_frame,
                textvariable=var,
                font=("Consolas", 11),
                wraplength=520,
                justify="left"
            ).grid(row=row, column=col + 1, sticky="nw", padx=5, pady=3)

        for c in range(4):
            info_frame.columnconfigure(c, weight=1)

    def _build_plot_panel(self):
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.fig = Figure(figsize=(12, 10), dpi=100)

        self.ax_dist = self.fig.add_subplot(511)
        self.ax_signal = self.fig.add_subplot(512, sharex=self.ax_dist)
        self.ax_status = self.fig.add_subplot(513, sharex=self.ax_dist)
        self.ax_valid = self.fig.add_subplot(514, sharex=self.ax_dist)
        self.ax_precision = self.fig.add_subplot(515, sharex=self.ax_dist)

        self.ax_dist.set_title("Distance (m)")
        self.ax_dist.set_ylabel("m")
        self.ax_dist.grid(True)

        self.ax_signal.set_title("Signal Strength")
        self.ax_signal.set_ylabel("signal")
        self.ax_signal.grid(True)

        self.ax_status.set_title("Status")
        self.ax_status.set_ylabel("status")
        self.ax_status.grid(True)

        self.ax_valid.set_title("Valid")
        self.ax_valid.set_ylabel("valid")
        self.ax_valid.grid(True)

        self.ax_precision.set_title("Range Precision (cm)")
        self.ax_precision.set_ylabel("cm")
        self.ax_precision.set_xlabel("Time (s)")
        self.ax_precision.grid(True)

        self.line_dist, = self.ax_dist.plot([], [], linewidth=2, label="Distance")
        self.line_signal, = self.ax_signal.plot([], [], linewidth=2, label="Signal")
        self.line_status, = self.ax_status.plot([], [], linewidth=2, label="Status")
        self.line_valid, = self.ax_valid.plot([], [], linewidth=2, label="Valid")
        self.line_precision, = self.ax_precision.plot([], [], linewidth=2, label="Precision")

        for ax in [self.ax_dist, self.ax_signal, self.ax_status, self.ax_valid, self.ax_precision]:
            ax.legend(loc="upper left")

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _set_info(self):
        m = self.monitor

        self.info_vars["port"].set(m.port)
        self.info_vars["baudrate"].set(str(m.baudrate))
        self.info_vars["query_module_id"].set(str(m.module_id))
        self.info_vars["query_interval"].set(f"{m.query_interval:.3f}")
        self.info_vars["last_request_packet"].set(m.last_request_packet)
        self.info_vars["frame_header"].set(f"0x{m.frame_header:02X}")
        self.info_vars["function_mark"].set(f"0x{m.function_mark:02X}")
        self.info_vars["reserved0"].set(f"0x{m.reserved0:02X}")
        self.info_vars["resp_module_id"].set(str(m.resp_module_id))
        self.info_vars["system_time_ms"].set(str(m.system_time_ms))
        self.info_vars["distance_mm"].set(f"{m.distance_mm:.1f}" if m.distance_mm >= 0 else "-")
        self.info_vars["distance_m"].set(f"{m.distance_m:.3f}" if m.distance_m >= 0 else "-")
        self.info_vars["dis_status"].set(str(m.dis_status))
        self.info_vars["valid"].set("1 (有效)" if m.valid == 1 else "0 (无效)")
        self.info_vars["signal_strength"].set(str(m.signal_strength))
        self.info_vars["range_precision_cm"].set(str(m.range_precision_cm))
        self.info_vars["checksum_ok"].set("True" if m.checksum_ok else "False")
        self.info_vars["tx_queries"].set(str(m.tx_queries))
        self.info_vars["rx_frames"].set(str(m.rx_frames))
        self.info_vars["timeout_frames"].set(str(m.timeout_frames))
        self.info_vars["bad_checksum_frames"].set(str(m.bad_checksum_frames))
        self.info_vars["last_update_str"].set(m.last_update_str)
        self.info_vars["last_raw_packet"].set(m.last_raw_packet)

    def _set_plots(self):
        m = self.monitor
        if len(m.time_data) == 0:
            return

        x = list(m.time_data)
        y_dist = list(m.distance_m_data)
        y_signal = list(m.signal_data)
        y_status = list(m.status_data)
        y_valid = list(m.valid_data)
        y_precision = list(m.precision_data)

        self.line_dist.set_xdata(x)
        self.line_dist.set_ydata(y_dist)

        self.line_signal.set_xdata(x)
        self.line_signal.set_ydata(y_signal)

        self.line_status.set_xdata(x)
        self.line_status.set_ydata(y_status)

        self.line_valid.set_xdata(x)
        self.line_valid.set_ydata(y_valid)

        self.line_precision.set_xdata(x)
        self.line_precision.set_ydata(y_precision)

        x_min = max(0, x[-1] - 15)
        x_max = x[-1] + 0.5
        self.ax_dist.set_xlim(x_min, x_max)

        dmin, dmax = min(y_dist), max(y_dist)
        dmargin = max(0.05, (dmax - dmin) * 0.1 if dmax > dmin else 0.1)
        self.ax_dist.set_ylim(max(0, dmin - dmargin), dmax + dmargin)

        smin, smax = min(y_signal), max(y_signal)
        smargin = max(10, (smax - smin) * 0.1 if smax > smin else 10)
        self.ax_signal.set_ylim(max(0, smin - smargin), smax + smargin)

        stmin, stmax = min(y_status), max(y_status)
        stmargin = max(1, (stmax - stmin) * 0.1 if stmax > stmin else 1)
        self.ax_status.set_ylim(stmin - stmargin, stmax + stmargin)

        self.ax_valid.set_ylim(-0.2, 1.2)

        pmin, pmax = min(y_precision), max(y_precision)
        pmargin = max(1, (pmax - pmin) * 0.1 if pmax > pmin else 1)
        self.ax_precision.set_ylim(max(0, pmin - pmargin), pmax + pmargin)

        self.canvas.draw_idle()

    def update_gui(self):
        self._set_info()
        self._set_plots()
        self.root.after(100, self.update_gui)


if __name__ == "__main__":
    monitor = LaserRangerQueryMonitor(
        port="/dev/ttyAMA1",
        baudrate=115200,
        module_id=0,
        query_interval=0.1,   # 100ms 查询一次
        history_len=500
    )
    monitor.start()

    root = tk.Tk()
    gui = MonitorGUI(root, monitor)

    def on_close():
        monitor.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()