import serial
import time
import threading

class LaserRanger:
    def __init__(self, port="/dev/ttyUSB1", baudrate=115200, timeout=1.0):
        """
        初始化激光测距模块
        :param port: 串口设备路径，注意树莓派上通常可能是 /dev/ttyUSB0 或 /dev/ttyUSB1 等
        :param baudrate: 波特率，默认通常是 115200 或 9600，具体需参考模块手册
        :param timeout: 串口读取超时时间
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.serial = None
        self.distance = -1.0
        self.running = False
        self._read_thread = None

        try:
            self.serial = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            print(f"[LaserRanger] 成功连接激光测距模块: {self.port} @ {self.baudrate}")
        except Exception as e:
            print(f"[LaserRanger] 无法连接激光测距模块: {e}")

    def start_continuous_reading(self):
        """启动后台线程持续读取测距数据"""
        if self.serial is None or not self.serial.is_open:
            print("[LaserRanger] 串口未打开，无法启动读取线程")
            return
            
        if self.running:
            return
            
        self.running = True
        self._read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self._read_thread.start()
        print("[LaserRanger] 测距数据读取线程已启动")

    def _read_loop(self):
        """后台读取循环，具体解析逻辑需要根据你的激光测距模块协议修改"""
        while self.running and self.serial and self.serial.is_open:
            try:
                if self.serial.in_waiting > 0:
                    # 假设模块发送的是以换行符结尾的ASCII字符串，如 "Distance: 123.4 mm\n"
                    # 或者是一个十六进制数据包。这里先写一个通用的字符串读取框架
                    line = self.serial.readline().decode('utf-8', errors='ignore').strip()
                    if line:
                        self._parse_data(line)
                else:
                    time.sleep(0.01)
            except Exception as e:
                print(f"[LaserRanger] 读取数据出错: {e}")
                time.sleep(0.1)

    def _parse_data(self, data_str):
        """
        解析激光模块返回的数据
        【注意】: 此处需要根据你购买的激光模块的通信协议进行重写！
        """
        # 这里仅作示例：假设返回的是纯数字字符串，代表毫米
        try:
            # 尝试提取数字部分
            import re
            match = re.search(r'\d+(\.\d+)?', data_str)
            if match:
                self.distance = float(match.group(0))
                # print(f"当前距离: {self.distance} mm")
        except ValueError:
            pass

    def get_distance(self):
        """获取最新读取到的距离值"""
        return self.distance

    def close(self):
        """关闭串口和线程"""
        self.running = False
        if self._read_thread is not None:
            self._read_thread.join(timeout=1.0)
            
        if self.serial is not None and self.serial.is_open:
            self.serial.close()
            print("[LaserRanger] 串口已关闭")

if __name__ == "__main__":
    # 简单测试代码
    ranger = LaserRanger(port="/dev/ttyUSB1", baudrate=115200)
    ranger.start_continuous_reading()
    
    try:
        for _ in range(10):
            print(f"当前获取到的距离: {ranger.get_distance()}")
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("测试中断")
    finally:
        ranger.close()