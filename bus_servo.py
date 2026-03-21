import serial
import time


class BusServo:
    # 初始化串口，总线舵机驱动板要求波特率为9600
    # moving_time为舵机移动的时间限制 单位ms
    def __init__(self, port='/dev/ttyAMA0', baudrate=9600, servo_num=2, servo_ids=[1, 2], moving_time=50):
        # 初始化串口 timeout = 1s
        self.serial = serial.Serial(port, baudrate=baudrate, timeout=1)
        # 舵机数量
        self.servo_num = servo_num
        # 舵机ids
        self.servo_ids = servo_ids
        
        # 设置的舵机角度目标值 由于舵机需要响应 所以未必是实际角度
        '''
        servo_angles_setting = [
            [1, 625],  # 舵机 ID 1, 角度值 625 居中
            [2, 250]   # 舵机 ID 2, 角度值 250 居中
        ]
        '''
        self.servo_angles_setting = [[1, 500], [2, 500]]  # 初始角度设置
        
        # 实时读取到的舵机角度 实际角度
        self.servo_angles_reading = [[1, 625], [2, 250]]  # 初始角度设置
        
#         # 角度限制
#         self.pan_min_angle = -90
#         self.pan_max_angle = 90
#         # 角度限制的index值
#         self.pan_min_angle_index = 250
#         self.pan_max_angle_index = 1000
#         
#         self.tilt_min_angle = -36
#         self.tilt_max_angle = 36
#         self.tilt_min_angle_index = 100
#         self.tilt_max_angle_index = 400



        # 角度限制
#         self.pan_min_angle = -30
#         self.pan_max_angle = 30
        
        self.pan_min_angle = -12
        self.pan_max_angle = 12
        
        
        # 角度限制的index值
        self.pan_min_angle_index = 500 - 50
        self.pan_max_angle_index = 500 + 50
        
        self.tilt_min_angle = -12
        self.tilt_max_angle = 12
        self.tilt_min_angle_index = 450
        self.tilt_max_angle_index = 550

    
        # 移动耗时设置
        self.moving_time = moving_time
        
        self.pan_bound_flag = False
        self.tilt_bound_flag = False
        
        # 判断舵机是否运动到目标角度的角度阈值 单位index 舵机角度为0-1000 index
        self.angle_equal_threshold = 2
        
        # 初始时舵机回中
        self.reset()
    
        
    
    # 对于单个舵机设置角度
    def set_angle(self, servo_id, angle):
        # 合法id
        if servo_id in self.servo_ids:
            # pan
            if servo_id == 1:
                self.pan_bound_flag = False
                # 限制角度
                if angle > self.pan_max_angle:
                    angle = self.pan_max_angle
                    self.pan_bound_flag = True
                    print('pan angle too big')
                elif angle < self.pan_min_angle:
                    angle = self.pan_min_angle
                    self.pan_bound_flag = True
                    print('pan angle too small')
                # 将角度进行映射为index
                angle_index = self.map(angle, self.pan_min_angle, self.pan_max_angle, \
                                       self.pan_min_angle_index, self.pan_max_angle_index)
                # 写入新角度
                self.servo_angles_setting[servo_id-1][1] = angle_index
                
                return self.pan_bound_flag
            
            # tilt
            if servo_id == 2:
                self.tilt_bound_flag = False
                if angle > self.tilt_max_angle:
                    angle = self.tilt_max_angle
                    self.tilt_bound_flag = True
                    print('tilt angle too big')
                elif angle < self.tilt_min_angle:
                    angle = self.tilt_min_angle
                    self.tilt_bound_flag = True
                    print('tilt angle too small')
                # 将角度进行映射为index
                angle_index = self.map(angle, self.tilt_min_angle, self.tilt_max_angle, \
                                       self.tilt_min_angle_index, self.tilt_max_angle_index)
                # 写入新角度
                self.servo_angles_setting[servo_id-1][1] = angle_index
                
                return self.tilt_bound_flag
            
        else:
            print('servo_id invalid!')
            
        
    
    
    # 所有舵机角度设置完毕后执行运动指令的函数
    def move_angle(self):
        # 构建并发送指令
        self.send_packet(self.build_packet_servos_move(self.servo_angles_setting))
        
        # 发送串口指令后等待舵机运动到位 等待时间等于指令中的舵机运动时间
        # sleep函数单位为秒，self.moving_time单位为ms，除以1000
        time.sleep(self.moving_time/1000)
        
#         # ----- 放弃判断移动到位 可能陷入死循环导致舵机响应延迟 ------  
#         # 以下代码用于在舵机进行大幅运动时 没有来得及运动到位 进行自动等待 保证运动到位
#         # 循环读取角度判断是否移动到位
#         while True:
#             # 读取角度
#             self.read_servos_angle()
#             # 判断是否移动到位 由于读取的角度值由于传感器限制有偏差 所以进行软判决 否则可能一直不相等进入死循环
#             if self.are_lists_soft_equal(self.servo_angles_setting, self.servo_angles_reading):
#                 print('target angle arrived')
#                 break
#             # 暂停后再次读取角度
#             time.sleep(self.moving_time/1000)
        

#         # ----- 放弃读取角度判断移动到位 开销过大 ------        
#         # 循环读取角度判断是否移动到位
#         while True:
#             # 读取角度
#             self.read_servos_angle()
#             # 判断是否移动到位 由于读取的角度值由于传感器限制有偏差 所以进行软判决 否则可能一直不相等进入死循环
#             if self.are_lists_soft_equal(self.servo_angles_setting, self.servo_angles_reading):
#                 print('target angle arrived')
#                 break
#             # 间隔0.05s 再次读取角度
#             time.sleep(0.05)
    
    
    def are_lists_soft_equal(self, list1, list2):
        if len(list1) != len(list2):
            return False
        for i in range(len(list1)):
            if abs(list1[i][1] - list2[i][1]) > self.angle_equal_threshold:
                return False
        return True
        
        
    # 读取所有舵机当前的实际角度
    def read_servos_angle(self):
        # 构建并发送指令
        self.send_packet(self.build_packet_angles_read(self.servo_ids))
        
        # 读取舵机控制板的回复
        '''
        格式：
        帧头 2 byte —— 0x55 0x55
        数据长度 1 byte —— 读取的舵机个数 × 3 + 3
        指令号 1 byte —— 21 即 0x15
        参数 1 + 3 * n —— 参数1：要读取的舵机个数 参数2：舵机ID 参数3：角度位置低8位 参数4：角度位置高8位 其余与参数234类似
        '''
        response_bytes = 2 + 1 + 1 + 1 + 3 * len(self.servo_ids)
        response = self.receive_data(response_bytes)
        # 截取参数部分
        prm_info = response[4:]
        # 读取总的舵机数
        num_read_servos = int(prm_info[0])
        # 读取每个舵机的舵机id和角度
        for i in range(num_read_servos):
            servo_id = prm_info[1 + i * 3]
            angle_low = prm_info[2 + i * 3]
            angle_high = prm_info[3 + i * 3]
            angle = int((angle_high << 8) | angle_low)
            self.servo_angles_reading[i] = [servo_id, angle]
            
        # print(self.servo_angles_reading)
        # print(self.servo_angles_setting)
        # return self.servo_angles_reading
         
        
    # 构建数据包 用于控制多个舵机移动到指定角度
    def build_packet_servos_move(self, servo_angles_setting):
        # 计算数据长度 = 控制舵机的个数 * 3 + 5
        length = len(servo_angles_setting) * 3 + 5
        # 创建空的 bytearray
        packet = bytearray()
        # 帧头 0x55 0x55
        packet.extend([0x55, 0x55])
        # 添加数据长度
        packet.append(length)
        # 添加指令
        CMD_SERVO_MOVE = 0x03
        packet.append(CMD_SERVO_MOVE)
        # ---- 添加参数 ----
        # 参数 1：要控制舵机的个数
        packet.append(len(servo_angles_setting))
        # 参数 2：时间低八位 单位ms
        packet.append(self.moving_time & 0xFF)  # 使用按位与运算符提取低8位
        # 参数 3：时间高八位 单位ms
        packet.append((self.moving_time >> 8) & 0xFF)  # 使用右移和按位与运算符提取高8位
        '''
        参数 4：舵机 ID 号
        参数 5：角度位置低八位
        参数 6：角度位置高八位
        参数......：格式与参数 4,5,6 相同，控制不同舵机的角度位置
        '''
        # 遍历舵机
        for servo_info in servo_angles_setting:
            servo_id = servo_info[0]
            servo_angle = servo_info[1]
            # 添加id指令
            packet.append(servo_id)
            # 添加角度指令
            packet.append(servo_angle & 0xFF)
            packet.append((servo_angle >> 8) & 0xFF) 
        
        # 打印指令 调试用
        # self.print_bytearray(packet)
        return packet
    
    
    # 构建数据包 用于读取多个舵机的角度位置值(默认0-1000) 
    def build_packet_angles_read(self, read_ids):
        # 计算数据长度 = 所需读取的舵机个数 + 3
        length = len(read_ids) + 3
        # 创建空的 bytearray
        packet = bytearray()
        # 帧头 0x55 0x55
        packet.extend([0x55, 0x55])
        # 添加数据长度
        packet.append(length)
        # 添加指令 
        CMD_SERVO_READ = 0x15
        packet.append(CMD_SERVO_READ)
        # ---- 添加参数 ----
        # 参数 1 要读取的舵机个数
        packet.append(len(read_ids))
        # 遍历需要进行角度读取的舵机id
        for read_id in read_ids:
            # 添加id参数
            packet.append(read_id)
        
        # 打印指令 调试用
        # self.print_bytearray(packet)
        return packet


    def print_bytearray(self, ba):
        hex_str = ' '.join(f'0x{byte:02x}' for byte in ba)
        print(hex_str)


    # 函数：发送数据包
    def send_packet(self, packet):
        self.serial.write(packet)
        
        
    def receive_data(self, num_bytes):
        # 从串口读取数据
        response = self.serial.read(num_bytes)
        return response
        
        
    def map(self, x, in_min, in_max, out_min, out_max):
        return round((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)


    def reset(self):
        # 舵机角度置零
        self.set_angle(1, 0)
        self.set_angle(2, 0)
        # 执行动作
        self.move_angle()
        print('reset complete!')
    
    
    # 结束 关闭串口
    def cleanup(self):
        self.serial.close()


if __name__ == '__main__':
    # 示例使用
    BusServo = BusServo()
    
    while True:
#         servo_id = int(input("Enter servo id: "))
#         angle = int(input("Enter angle, pan id = 1 (-90 to 90) ; tilt = 2 (-36 to 36): "))
#         BusServo.set_angle(servo_id, angle)
#         BusServo.move_angle()
#         if input('按q退出') == 'q':
#             break


        servo_id = int(input("Enter servo id: "))
        angle = int(input("Enter angle, pan id = 1 (-90 to 90) ; tilt = 2 (-36 to 36): "))
        BusServo.set_angle(servo_id, angle)
        BusServo.move_angle()
        if input('按q退出') == 'q':
            break
        
    BusServo.cleanup()
