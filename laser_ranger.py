#coding: UTF-8
import serial 
import time
import threading
import tkinter as tk
from tkinter import ttk
import collections

# 为了绘图导入 matplotlib
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

class LaserRanger:
    def __init__(self, port='/dev/ttyAMA2', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.serial = None
        self.running = False
        self.read_thread = None
        
        # 协议常量
        self.TOF_length = 16
        self.TOF_header = (87, 0, 255)
        
        # 数据存储
        self.distance_mm = -1.0
        self.status = 0
        self.signal = 0
        
        # 绘图用数据队列（最多存最近100个点）
        self.time_data = collections.deque(maxlen=100)
        self.dist_data = collections.deque(maxlen=100)
        self.start_time = time.time()

        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=0.5)
            self.serial.flushInput()
            print(f"[LaserRanger] 已连接串口 {self.port}")
        except Exception as e:
            print(f"[LaserRanger] 串口连接失败: {e}")

    def verify_checksum(self, data, length):
        tof_check = 0
        for k in range(0, length - 1):
            tof_check += data[k]
        tof_check = tof_check % 256
        return tof_check == data[length - 1]

    def start(self):
        if self.serial is None or not self.serial.is_open:
            return
        self.running = True
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()

    def stop(self):
        self.running = False
        if self.read_thread is not None:
            self.read_thread.join(timeout=1.0)
        if self.serial is not None and self.serial.is_open:
            self.serial.close()

    def _read_loop(self):
        # 缓存区，用于拼接不完整的数据包
        buffer = []
        while self.running and self.serial and self.serial.is_open:
            try:
                # 每次尽可能多读
                waiting = self.serial.in_waiting
                if waiting > 0:
                    raw_data = self.serial.read(waiting)
                    buffer.extend(list(raw_data))
                    
                    # 寻找包头 87 0 255
                    while len(buffer) >= self.TOF_length:
                        # 查找包头位置
                        header_idx = -1
                        for i in range(len(buffer) - 2):
                            if buffer[i] == self.TOF_header[0] and \
                               buffer[i+1] == self.TOF_header[1] and \
                               buffer[i+2] == self.TOF_header[2]:
                                header_idx = i
                                break
                                
                        if header_idx == -1:
                            # 没找到包头，保留最后两个字节（可能是残缺的包头），其他丢弃
                            buffer = buffer[-2:]
                            break
                            
                        # 如果包头后的数据不够一个完整包，等待下次读取
                        if len(buffer) - header_idx < self.TOF_length:
                            buffer = buffer[header_idx:]
                            break
                            
                        # 提取一个完整包
                        packet = buffer[header_idx : header_idx + self.TOF_length]
                        # 移除已经处理过的数据（包括包头之前的无用数据）
                        buffer = buffer[header_idx + self.TOF_length :]
                        
                        if self.verify_checksum(packet, self.TOF_length):
                            signal = packet[12] | (packet[13] << 8)
                            if signal == 0:
                                # print("Out of range!")
                                pass
                            else:
                                distance = packet[8] | (packet[9] << 8) | (packet[10] << 16)
                                self.distance_mm = float(distance)
                                self.status = packet[11]
                                self.signal = signal
                                
                                # 记录绘图数据
                                current_t = time.time() - self.start_time
                                self.time_data.append(current_t)
                                self.dist_data.append(self.distance_mm)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[LaserRanger] 读取错误: {e}")
                time.sleep(0.1)

class LaserGUI:
    def __init__(self, root, ranger):
        self.root = root
        self.ranger = ranger
        self.root.title("激光测距实时监控")
        self.root.geometry("800x600")

        # 顶部状态显示
        self.info_var = tk.StringVar()
        self.info_var.set("等待数据...")
        self.lbl_info = ttk.Label(self.root, textvariable=self.info_var, font=("Arial", 16, "bold"))
        self.lbl_info.pack(pady=10)

        # 绘图区域
        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Real-time Distance")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Distance (mm)")
        self.ax.grid(True)
        
        self.line, = self.ax.plot([], [], 'b-', linewidth=2)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 定时器更新
        self.update_plot()

    def update_plot(self):
        if len(self.ranger.time_data) > 0:
            times = list(self.ranger.time_data)
            dists = list(self.ranger.dist_data)
            
            # 更新状态文字
            latest_dist = dists[-1]
            self.info_var.set(f"当前距离: {latest_dist:.1f} mm | 信号强度: {self.ranger.signal}")
            
            # 更新曲线
            self.line.set_xdata(times)
            self.line.set_ydata(dists)
            
            # 动态调整坐标轴范围
            self.ax.set_xlim(max(0, times[-1] - 10), times[-1] + 1) # 显示最近10秒
            
            min_d = min(dists)
            max_d = max(dists)
            margin = max(10, (max_d - min_d) * 0.1)
            self.ax.set_ylim(max(0, min_d - margin), max_d + margin)
            
            self.canvas.draw_idle()
            
        self.root.after(50, self.update_plot) # 50ms 刷新一次

if __name__ == "__main__":
    ranger = LaserRanger(port='/dev/ttyAMA2', baudrate=115200)
    ranger.start()

    root = tk.Tk()
    app = LaserGUI(root, ranger)
    
    def on_closing():
        ranger.stop()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()
          
         
    
        
    





