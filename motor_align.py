import serial
import binascii
import time
import struct


class Motor:
    def __init__(self, serial_port="/dev/ttyAMA0", max_speed=36000, motor_id=1):
        # 初始化串口
        self.ser = serial.Serial(serial_port, 115200, timeout=1)
        self.max_speed = max_speed
        self.motor_id = motor_id

        self.current_angle = 0

    def calculate_checksum(self, cmd):
        """
        Calculate checksum.
        :param cmd: Byte data to be sent (excluding checksum).
        :return: Checksum byte.
        """
        return sum(cmd) & 0xFF

    def send_command(self, angle_control):
        """
        Send the multi-turn position control command.
        :param angle_control: Target angle in hundredths of degrees.
        :param max_speed: Maximum speed in hundredths of degrees per second.
        :param motor_id: Motor ID (1 to 32).
        """

        self.current_angle = angle_control

        angle_control = int(angle_control * 100)

        frame_header = 0x3E
        command = 0xA4
        length = 0x0C

        # Convert control values to bytes
        angle_control_bytes = list(angle_control.to_bytes(8, byteorder='little', signed=True))
        max_speed_bytes = list(self.max_speed.to_bytes(4, byteorder='little', signed=False))

        # Build command frame
        command_frame = [frame_header, command, self.motor_id, length]
        checksum = self.calculate_checksum(command_frame)  # Fix method call here
        command_frame.append(checksum)

        # Build data frame
        data_frame = angle_control_bytes + max_speed_bytes
        checksum_data = self.calculate_checksum(data_frame)  # Fix method call here
        data_frame.append(checksum_data)

        # Combine command frame and data frame
        full_frame = bytes(command_frame + data_frame)

        # Send the command via UART
        self.ser.write(full_frame)
        # print(f"Sent: {binascii.hexlify(full_frame).decode()}")

    def read_reply(self):
        """
        Read and display motor's response.
        The response contains the following:
        - Motor temperature (int8_t, 1°C/LSB)
        - Torque current or output power (int16_t, -33A to 33A for torque current or -1000 to 1000 for power)
        - Motor speed (int16_t, 1dps/LSB)
        - Encoder position (uint16_t, 14-bit encoder range)
        """
        time.sleep(0.005)  # Wait for motor response
        response = self.ser.read(self.ser.in_waiting)  # Read all available bytes

        '''
        1.电机温度 temperature（int8_t 类型，1℃/LSB）。
        2. MF、MG 的转矩电流值 iq 或 MS 的输出功率值 power。iq 为 int16_t 类型，
        范围-2048~2048，对应实际转矩电流范围-33A~33A；power 为 int16_t 类型，
        范围-1000~1000。
        3. 电机转速 speed（int16_t 类型，1dps/LSB）。
        4. 编码器位置值 encoder（uint16_t 类型，14bit 编码器的数值范围 0~16383）。
        '''
        
        if response:
            print(f"Received: {binascii.hexlify(response).decode()}")
            # 帧命令 5 byte  帧数据 8 byte
            if len(response) >= 5 + 8:
                # Parse header and frame information
                header, cmd, motor_id_resp, data_length, checksum_resp = response[:5]
                # 0xa4是多圈位置闭环指令2 的cmd代号
                if header == 0x3E and cmd == 0xa4 and data_length == 0x07:
                    # Extract data bytes
                    temperature = struct.unpack('<b', response[5:6])[0]  # 1 byte (int8_t)
                    iq_or_power_low = struct.unpack('<B', response[6:7])[0]  # Low byte of iq or power
                    iq_or_power_high = struct.unpack('<B', response[7:8])[0]  # High byte of iq or power
                    iq_or_power = struct.unpack('<h', bytes([iq_or_power_low, iq_or_power_high]))[0]  # 2 bytes (int16_t)

                    speed_low = struct.unpack('<B', response[8:9])[0]  # Low byte of speed
                    speed_high = struct.unpack('<B', response[9:10])[0]  # High byte of speed
                    speed = struct.unpack('<h', bytes([speed_low, speed_high]))[0]  # 2 bytes (int16_t)

                    encoder_low = struct.unpack('<B', response[10:11])[0]  # Low byte of encoder
                    encoder_high = struct.unpack('<B', response[11:12])[0]  # High byte of encoder
                    encoder = struct.unpack('<H', bytes([encoder_low, encoder_high]))[0]  # 2 bytes (uint16_t)

                    print(f"Temperature: {temperature} °C")
                    print(f"Torque Current/Power: {iq_or_power} (value in A or W)")
                    print(f"Speed: {speed} dps")
                    print(f"Encoder Position: {encoder}")
                else:
                    print("Invalid response frame.")
            else:
                print("Response too short to parse.")
        else:
            print("No response received.")

    def close_motor(self):
        self.ser.close()
        