#-*- coding: UTF-8 -*-

# 调用必需库
from multiprocessing import Manager
from multiprocessing import Process
from PID import PID

import signal
import time
import sys
import cv2
from picamera2 import Picamera2, Preview
import numpy as np
from PIL import Image
from collections import deque
from gpiozero import LED
from pprint import *
from libcamera import controls

from motor_align import Motor

# 创建cv2 Trackbar的回调函数
def on_trackbar(val):
    pass


def circle_obj(objX, objY, centerX, centerY, circle_exist_flag):
  # 定义 picamera2 对象
    picam2 = Picamera2()

    # 传感器配置
    # pprint(picam2.sensor_modes)
    '''
    {'bit_depth': 12,
    'crop_limits': (0, 440, 4056, 2160),
    'exposure_limits': (60, 674181621, None),
    'format': SRGGB12_CSI2P,
    'fps': 50.03,
    'size': (2028, 1080),
    'unpacked': 'SRGGB12'}
    '''

    # 空预览 用于驱动相机
    picam2.start_preview(Preview.NULL)

    # video 配置，still 配置像素高但采集慢
    framerate = 30
    # framerate = 1000000 / frameduration
    frameduration = int(1000000 / framerate)
    # picamera2 doc 设置 framerate
    config = picam2.create_video_configuration(controls={"FrameDurationLimits": (frameduration, frameduration)})
    
    # 设置格式 原始是XBGR8888
    # opencv默认RGB888
    # RGB888  (height, width, 3)  Each pixel is laid out as [B, G, R].
    config["main"]["format"] = "RGB888"

    # 图像大小
    config['main']['size'] = (640, 640)
    # picam2 自动选择最优的分辨率
    picam2.align_configuration(config)
    # 显示最终的分辨率(1024,1080)
    # pprint(config['main'])
    
    # 执行配置
    picam2.configure(config)
    # 打印配置
    pprint(picam2.camera_configuration())

    # 设置 摄像头的自动曝光 和 自动白平衡
    picam2.set_controls({"AeEnable": True, 'AwbEnable': True})

    # 打印摄像头control变量配置
    camera_control_settings = picam2.camera_controls
    pprint(camera_control_settings)

    # 启动相机
    picam2.start()
    
    # 等待1秒
    time.sleep(1)

    # 创建GUI窗口
    cv2.namedWindow('Circle Tracking')

    # 曝光值和增益值
    '''
    'ExposureValue': (-8.0, 8.0, 0.0),
    'AnalogueGain': (1.0, 22.2608699798584, None)
    '''
    # 设置 ExposureValue 最大值为 8 + 8 = 16，默认为8
    cv2.createTrackbar('ExposureValue', 'Circle Tracking', 8, int(camera_control_settings['ExposureValue'][1]) + 8, on_trackbar)
    # 设置 AnalogueGain 最大值为 22，默认为11
    cv2.createTrackbar('AnalogueGain', 'Circle Tracking', 11, int(camera_control_settings['AnalogueGain'][1]), on_trackbar)

    # 创建滑动条
    cv2.createTrackbar('ksize', 'Circle Tracking', 5, 10, on_trackbar)
    cv2.createTrackbar('minDist', 'Circle Tracking', 100, 200, on_trackbar)
    cv2.createTrackbar('param1', 'Circle Tracking', 400, 500, on_trackbar)
    cv2.createTrackbar('param2', 'Circle Tracking', 40, 100, on_trackbar)
    cv2.createTrackbar('minRadius', 'Circle Tracking', 30, 150, on_trackbar)
    cv2.createTrackbar('maxRadius', 'Circle Tracking', 50, 150, on_trackbar)
    
    # 坐标偏移量
    cv2.createTrackbar('x_bias', 'Circle Tracking', 155, 500, on_trackbar)
    cv2.createTrackbar('y_bias', 'Circle Tracking', 100, 500, on_trackbar)
    
    # 判断是否进行追踪的flag
    cv2.createTrackbar('Track Flag', 'Circle Tracking', 1, 1, on_trackbar)

    try:
        # 初始化变量 记录上一个time step的曝光和增益
        last_ExposureValue = cv2.getTrackbarPos('ExposureValue', 'Circle Tracking') - 8
        last_AnalogueGain = cv2.getTrackbarPos('AnalogueGain', 'Circle Tracking')
        
        # 创建一个队列用于存储目标位置（后续进行队列内平滑）
        target_positions = deque(maxlen=1)
        
        # 初始化计时器
        start_time = time.time()
        count = 0
        fps = 0

        #进入循环
        while True:
            # 获取Trackbar的参数值
            ExposureValue = cv2.getTrackbarPos('ExposureValue', 'Circle Tracking') - 8
            AnalogueGain = cv2.getTrackbarPos('AnalogueGain', 'Circle Tracking')
            
            ksize = cv2.getTrackbarPos('ksize', 'Circle Tracking')
            # 高斯滤波核只能是奇数
            if ksize % 2 == 0:
                ksize += 1
            
            minDist = cv2.getTrackbarPos('minDist', 'Circle Tracking')
            param1 = cv2.getTrackbarPos('param1', 'Circle Tracking')
            param2 = cv2.getTrackbarPos('param2', 'Circle Tracking')
            minRadius = cv2.getTrackbarPos('minRadius', 'Circle Tracking')
            maxRadius = cv2.getTrackbarPos('maxRadius', 'Circle Tracking')
            
            # 减去最大值的一半 正负对半分布
            x_bias = cv2.getTrackbarPos('x_bias', 'Circle Tracking')
            x_bias = x_bias - 150
            y_bias = cv2.getTrackbarPos('y_bias', 'Circle Tracking')
            y_bias = y_bias - 150
            
            # 对准标志
            Track_Flag = cv2.getTrackbarPos('Track Flag', 'Circle Tracking')
            
            # 修改了曝光或者增益
            if last_ExposureValue != ExposureValue or last_AnalogueGain != AnalogueGain:
                picam2.set_controls({"AnalogueGain": AnalogueGain, "ExposureValue": ExposureValue})
                # 更新曝光和增益
                last_ExposureValue = ExposureValue
                last_AnalogueGain = AnalogueGain
                print('change camera settings!')
            else:
                pass
            
            image = picam2.capture_array()
            # 水平翻转 mirror effect
            # image = cv2.flip(image, 1)
            
            # 找到图像中心
            (H, W) = image.shape[:2]
            centerX.value = W // 2
            centerY.value = H // 2

            # 画出图像中心点
            cv2.circle(image, (centerX.value, centerY.value), 2, (255, 0, 255), -1)
            
            # 分离通道 [B, G, R]
            red_channel = image[:, :, 2]
            green_channel = image[:, :, 1]
            blue_channel = image[:, :, 0]
        
            sigmaX = 1
            blurred = cv2.GaussianBlur(green_channel, (ksize, ksize), sigmaX)
        
            # 调用霍夫圆检测函数，进行初始检测
            '''
            dp：累加器分辨率与图像分辨率的比值。默认值为1，表示两者相等。较小的值可以提高圆的检测速度，但可能会导致更多的误检测。
            minDist：检测到的圆之间的最小距离。默认值为0，表示不对圆之间的距离进行限制。你可以根据实际情况设置一个合适的值来避免检测到过于接近的圆。
            param1：边缘检测的阈值。较大的值可以过滤掉较弱的边缘，减少误检测。你可以根据图像的特性进行调整。
            param2：累加器阈值，用于确定圆的检测结果。较小的值会导致更多的圆被检测到，较大的值会减少圆的数量，可以用于过滤掉累加器值较小的圆。
            minRadius和maxRadius：圆的最小半径和最大半径。你可以根据目标圆的大小范围设置合适的值，限制圆的大小。
            '''
            circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=minDist,
                                       param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)
            
            
            if circles is not None:
                # 找到圆形 修改标志
                circle_exist_flag.value = True
                
                circles = np.round(circles).astype(int)
                '''
                    circles数组的形状是 (1, N, 3)，其中 N 是检测到的圆的数量。
                    每个圆用三个值表示：(x, y, radius)，分别表示圆心的 x 坐标、y 坐标和圆的半径。
                    for i in circles[0,:] 这一行代码是遍历 circles 数组中的每个圆，其中 i 表示当前的圆。
                '''
                for (x, y, r) in circles[0]:
                    # 绘制圆心 和 圆周
                    cv2.circle(image, (x, y), 1, (0, 0, 255), 2)
                    cv2.circle(image, (x, y), r, (0, 0, 255), 1)

                # 人为规则：
                # 取第一个圆形标志物的位置作为跟踪目标
                target_circle = circles[0][0]
                target_center = (target_circle[0], target_circle[1])
                target_radius = target_circle[2]
                
                # 将目标位置添加到队列中
                target_positions.append((target_center, target_radius))
                
                # 在队列中计算中心位置的平均值
                avg_center_x = sum(pos[0][0] for pos in target_positions) / len(target_positions)
                avg_center_y = sum(pos[0][1] for pos in target_positions) / len(target_positions)
                avg_center = (int(avg_center_x), int(avg_center_y))
                
                # 在队列中计算半径的平均值
                avg_radius = int(sum(pos[1] for pos in target_positions) / len(target_positions))
                
                # 输出平均位置
                # print("Average Center:", avg_center)
                # print("Average Radius:", avg_radius)
                
                # 当前跟踪的圆绘制
                # 字体设置
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                font_color = (0, 255, 0)
                font_thickness = 2
                font_pos = (avg_center[0] - 30, avg_center[1] - 10)
                cv2.putText(image, 'tracking', font_pos, font, font_scale, font_color, font_thickness)
                cv2.circle(image, avg_center, 1, (0, 255, 0), 3)  # 圆心
                cv2.circle(image, avg_center, avg_radius, (0, 255, 0), 1)  # 轮廓
                
                # 开始追踪
                if Track_Flag == 1:
                    # 设置追踪目标坐标
                    objX.value = avg_center[0] + x_bias
                    objY.value = avg_center[1] + y_bias
                    objRadius = avg_radius
                else:
                    pass
            
            else:
                # 找不到圆形
                circle_exist_flag.value = False

            # 不进行追踪
            if Track_Flag == 0:
                # 设置追踪目标坐标为图像中心坐标 即PID此时的error输入是0
                objX.value = centerX.value
                objY.value = centerY.value
                objRadius = 1
            
            # 画出追踪目标点的位置
            cv2.circle(image, (objX.value, objY.value), 2, (255, 255, 255), -1)
            
            # 缩放图像
            target_width = 640
            target_height = 640 

            # 放缩尺寸
            image_scale = cv2.resize(image, (target_width, target_height))
            
            # RGB单通道
            # r_scale = cv2.resize(red_channel, (target_width, target_height))
            # g_scale = cv2.resize(green_channel, (target_width, target_height))
            # b_scale = cv2.resize(blue_channel, (target_width, target_height))

            # combined_image = cv2.hconcat([r_scale, g_scale, b_scale])
            
            # 统计循环次数
            count += 1

            # 检查是否已经过了1秒
            current_time = time.time()
            elapsed_time = current_time - start_time
            if elapsed_time >= 1.0:
                # 计算FPS
                fps = count / elapsed_time
                # 重置计时器和计数器
                start_time = current_time
                count = 0
            
            # 输出每秒的循环次数
            cv2.putText(image_scale, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
            # 显示处理后的图像
            cv2.imshow('Circle Tracking', image_scale)

            cv2.waitKey(1)
            
    except KeyboardInterrupt:
        # 关闭摄像头
        picam2.stop()
        # 关闭所有窗口
        cv2.destroyAllWindows()
        # print('-------- camera closed! --------')
        print('circle_obj process closed')
   
   
# panAngle, tiltAngle = output   bound_flag可以是pan和tilt的
def pid_process(output, p, i, d, PID_update_interval_sec, objCoord, centerCoord, circle_exist_flag, bound_flag):
    # 创建一个PID类的对象并初始化
    pid = PID(p.value, i.value, d.value, output_bound_low=-12, output_bound_high=12)

    error_queue = deque(maxlen=1)
    
    try:
        time.sleep(1.5)
        print('pid_process wait for camera init')
        
        # 进入循环
        while True:
            
            # 计算误差
            error = centerCoord.value - objCoord.value
            
            error_queue.append(error)
            avg_error = sum(error_queue) / len(error_queue)
            
            print('avg_error' + str(avg_error))
            
            # 更新输出值,当error小于50时，误差设为0，以避免云台不停运行。
            if abs(avg_error) < error_threshold.value:
                error = 0
            
            # PID sleep 时间等于舵机的moving time
            # 假如找不到圆形标志物 target_flag 增加积分限制 防止无目的漂移
            output.value = pid.update(error, sleep=PID_update_interval_sec.value)
            
            # output.value = pd.update(error, sleep=PID_update_interval_sec.value)
    
    except KeyboardInterrupt:
    # except Exception as e:
    # 捕获所有的异常
        print('PID process closed')


def set_servos(panAngle, tiltAngle, PID_max_delta_angle_cut, pan_bound_flag, tilt_bound_flag):
    # id=1为pan id=2为tilt
    pan_id = 1

    try:
        time.sleep(1.5)
        print('set_servos wait for camera init')
        #进入循环
        
        while True:
            # 偏角变号
            # 因为舵机的定义的角度和pid算出来的反过来 算一个bug
            pan_angle_new = 1 * panAngle.value
            
            # 返回当前角度
            pan_angle_old = BusServo.current_angle
            
            # 计算角度差值
            pan_diff = abs(pan_angle_old - pan_angle_new)
         
            # 对于新旧角度差值进行判断 差值超出舵机响应速度后进行clip 防止舵机未移动到位 导致过冲摇摆
            # if pan_diff > PID_max_delta_angle_cut.value:
            if pan_diff > 0.01:
                pan_angle_new = pan_angle_old - max_delta_angle_cut if pan_angle_old > pan_angle_new else pan_angle_old + max_delta_angle_cut
            
            # 执行动作
            BusServo.send_command(pan_angle_new)

            print(pan_angle_new)

            time.sleep(servo_update_interval_sec)
    
    except KeyboardInterrupt:
    # except Exception as e:
    # 捕获所有的异常
        # 关闭舵机
        BusServo.close_motor()
        print('set_servos process closed')
        

# 启动主程序
if __name__ == "__main__":
    # 舵机移动时间 单位ms
    servo_update_interval_ms = int(1000/100)     # 30Hz
    servo_update_interval_sec = servo_update_interval_ms / 1000
    
    # 由于舵机速度有限 需要对于输出的角度进行clip
    # 0.22sec = 60度 移动1度 = 0.22/60 = 0.0036666sec
    print('servo_update_interval_sec: ' + str(servo_update_interval_sec))
    # 保护余量 95%
    max_delta_angle_cut = 0.95 * servo_update_interval_sec * 30     # 电机是360dps
    print('max_delta_angle_cut: ' + str(max_delta_angle_cut))
    
    processes = []
    
    # 全局变量 舵机
    BusServo = Motor("/dev/ttyAMA0", max_speed=3000, motor_id=1)
    
    try:
        # 启动多进程变量管理
        with Manager() as manager:  # 相当于manager=Manager(),with as 语句操作上下文管理器（context manager），它能够帮助我们自动分配并且释放资源。
            # 为图像中心坐标赋初值
            centerX = manager.Value("i", 0)  # "i"即为整型integer
            centerY = manager.Value("i", 0)

            # 为圆形中心坐标赋初值
            objX = manager.Value("i", 0)
            objY = manager.Value("i", 0)

            # panAngle和tiltAngle分别是两个舵机的PID控制输出量
            panAngle = manager.Value("i", 0)
            tiltAngle = manager.Value("i", 0)

            # 设置一级舵机的PID参数
            panP = manager.Value("f", 0.0075)  # "f"即为浮点型float
            panI = manager.Value("f", 0.025)
            panD = manager.Value("f", 0.000005)
            # 设置二级舵机的PID参
            tiltP = manager.Value("f", 0.01)
            tiltI = manager.Value("f", 0.02)
            tiltD = manager.Value("f", 0.000005)

            # PID结束阈值偏差
            error_threshold = manager.Value('f', 3)
          
            
            # 是否存在圆形标志物体
            circle_exist_flag = manager.Value('b', False)
            
            # 是否运动到边界
            pan_bound_flag = manager.Value('b', False)
            tilt_bound_flag = manager.Value('b', False)
            
            # PID更新时间间隔 单位sec
            PID_update_interval_sec = manager.Value('f', servo_update_interval_sec)
            
            # PID update 角度限制
            PID_max_delta_angle_cut = manager.Value('f', max_delta_angle_cut)

            # 创建4个独立进程
            # 1. objectCenter  - 探测人脸
            # 2. panning       - 对一级舵机进行PID控制，控制偏航角
            # 3. tilting       - 对二级舵机进行PID控制，控制俯仰角
            # 4. setServos     - 根据PID控制的输出驱动舵机

            processObjectCenter = Process(target=circle_obj,
                                          args=(objX, objY, centerX, centerY, circle_exist_flag))
            processPanning = Process(target=pid_process,
                                     args=(panAngle, panP, panI, panD, PID_update_interval_sec, objX, centerX, circle_exist_flag, pan_bound_flag))
            # processTilting = Process(target=pid_process,
            #                          args=(tiltAngle, tiltP, tiltI, tiltD, PID_update_interval_sec, objY, centerY, circle_exist_flag, tilt_bound_flag))
            processSetServos = Process(target=set_servos,
                                       args=(panAngle, tiltAngle, PID_max_delta_angle_cut, pan_bound_flag, tilt_bound_flag))
            
            processes = [processObjectCenter, processPanning, processSetServos]
            
            # 开启4个进程
            for p in processes:
                p.start()
                
            # 添加4个进程
            for p in processes:
                p.join()

    except KeyboardInterrupt:
        # 捕获键盘中断信号（Ctrl+C）
        pass
    finally:
        time.sleep(0.1)
        # 关闭所有进程
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join()
                print('process terminating')
            else:
                print('process has already terminated')
