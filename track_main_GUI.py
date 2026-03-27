import time
import os
import json
import math
import threading
import queue
import sys
import importlib.util
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import cv2
import numpy as np
from PIL import Image, ImageTk
from picamera2 import Picamera2, Preview

from PID import PID
from laser_ranger_passive import LaserRangerQueryMonitor
from laser_ranger_setting import configure_laser_module
from imu import IMUReader


def _load_module(name, path):
    # 避免重复加载同一模块，特别是覆盖原有的模块
    if name in sys.modules:
        return sys.modules[name]
        
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # 必须把 module 注册到 sys.modules 中，否则 dataclasses 在解析时
    # 通过 sys.modules.get(cls.__module__) 获取模块时会得到 None，从而引发 'NoneType' object has no attribute '__dict__' 错误
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Kalman2D:
    def __init__(self):
        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.4
        self.kf.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_estimate = None

    def reset(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True
        self.last_estimate = (float(x), float(y))

    def update_params(self, process_noise, measurement_noise):
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * float(process_noise)
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * float(measurement_noise)

    def update(self, measurement):
        if measurement is not None and not self.initialized:
            self.reset(measurement[0], measurement[1])
        if measurement is None and not self.initialized:
            return None
        if measurement is None:
            if self.last_estimate is None:
                prediction = self.kf.predict()
                estimate = prediction
                self.last_estimate = (float(estimate[0]), float(estimate[1]))
            else:
                self.kf.statePost = np.array(
                    [[self.last_estimate[0]], [self.last_estimate[1]], [0], [0]],
                    dtype=np.float32,
                )
                estimate = self.kf.statePost
                prediction = estimate
            return float(estimate[0]), float(estimate[1]), float(prediction[0]), float(prediction[1])
        prediction = self.kf.predict()
        measured = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
        estimate = self.kf.correct(measured)
        self.last_estimate = (float(estimate[0]), float(estimate[1]))
        return float(estimate[0]), float(estimate[1]), float(prediction[0]), float(prediction[1])


class BoardServoAdapter:
    def __init__(self, driver, servo_ids, moving_time=50):
        self.driver = driver
        self.servo_ids = list(servo_ids)
        self.moving_time = int(moving_time)
        self._pending_positions = {sid: 500 for sid in self.servo_ids}

    def set_angles(self, angles):
        for servo_id, angle in angles:
            angle_index = 500 + float(angle) / 0.24
            angle_index = max(0, min(1000, int(round(angle_index))))
            self._pending_positions[int(servo_id)] = angle_index

    def move_angle(self, wait=True):
        positions = [(sid, self._pending_positions.get(sid, 500)) for sid in self.servo_ids]
        self.driver.move_servos(positions, self.moving_time)
        if wait:
            time.sleep(self.moving_time / 1000.0)

    def read_positions(self):
        result = self.driver.read_servo_positions(self.servo_ids)
        return {item.servo_id: item.position for item in result}

    def read_voltage_mv(self):
        return self.driver.get_battery_voltage_mv()

    def cleanup(self):
        self.driver.close()


class BrushlessDualServoAdapter:
    def __init__(
        self,
        motor_config_cls,
        motor_cls,
        pan_id,
        tilt_id,
        pan_dev,
        tilt_dev,
        pan_baudrate,
        tilt_baudrate,
        pan_txden,
        tilt_txden,
        pan_direction_sign,
        tilt_direction_sign,
        pan_speed_dps,
        tilt_speed_dps,
        pan_min_deg,
        pan_max_deg,
        tilt_min_deg,
        tilt_max_deg,
    ):
        self.pan_id = int(pan_id)
        self.tilt_id = int(tilt_id)
        self.pan_speed_dps = float(pan_speed_dps)
        self.tilt_speed_dps = float(tilt_speed_dps)
        self.pan_motor = motor_cls(
            motor_config_cls(
                name="yaw",
                motor_id=int(pan_id),
                dev=str(pan_dev),
                baudrate=int(pan_baudrate),
                txden_pin=int(pan_txden),
                direction_sign=int(pan_direction_sign),
                default_speed_dps=float(pan_speed_dps),
                min_deg=float(pan_min_deg),
                max_deg=float(pan_max_deg),
            )
        )
        self.tilt_motor = motor_cls(
            motor_config_cls(
                name="pitch",
                motor_id=int(tilt_id),
                dev=str(tilt_dev),
                baudrate=int(tilt_baudrate),
                txden_pin=int(tilt_txden),
                direction_sign=int(tilt_direction_sign),
                default_speed_dps=float(tilt_speed_dps),
                min_deg=float(tilt_min_deg),
                max_deg=float(tilt_max_deg),
            )
        )
        self._pending_pan_deg = 0.0
        self._pending_tilt_deg = 0.0
        self.pan_motor.motor_run()
        time.sleep(0.03)
        self.tilt_motor.motor_run()

    def set_angles(self, angles):
        if self.pan_id == self.tilt_id:
            ordered = list(angles)
            if len(ordered) >= 1:
                self._pending_pan_deg = float(ordered[0][1])
            if len(ordered) >= 2:
                self._pending_tilt_deg = float(ordered[1][1])
            return
        for servo_id, angle in angles:
            sid = int(servo_id)
            if sid == self.pan_id:
                self._pending_pan_deg = float(angle)
            elif sid == self.tilt_id:
                self._pending_tilt_deg = float(angle)

    def move_angle(self, wait=True):
        self.pan_motor.move_to_deg(self._pending_pan_deg, max_speed_dps=self.pan_speed_dps)
        time.sleep(0.005)
        self.tilt_motor.move_to_deg(self._pending_tilt_deg, max_speed_dps=self.tilt_speed_dps)
        if wait:
            time.sleep(0.01)

    def read_servos_angle(self):
        pan = self.pan_motor.read_multi_turn_angle_deg()
        tilt = self.tilt_motor.read_multi_turn_angle_deg()
        return [(self.pan_id, pan), (self.tilt_id, tilt)]

    def cleanup(self):
        try:
            self.pan_motor.motor_stop()
        except Exception:
            pass
        try:
            self.tilt_motor.motor_stop()
        except Exception:
            pass
        try:
            self.pan_motor.close()
        except Exception:
            pass
        try:
            self.tilt_motor.close()
        except Exception:
            pass


class CircleTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("圆形追踪器")
        cv2.setUseOptimized(True)
        cpu_count = os.cpu_count() or 1
        cv2.setNumThreads(max(1, cpu_count - 1))
        self.running = False
        self.tracking_active = False
        self.picam2 = None
        self.servo = None
        self.laser_ranger = None  # 激光测距模块
        self.imu = None
        self.imu_zero_pitch = 0.0
        self.imu_zero_yaw = 0.0
        self.latest_imu = None
        self.worker_thread = None
        self.detect_thread = None
        self.stop_event = threading.Event()
        self.detect_stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.settings_lock = threading.Lock()
        self.detect_lock = threading.Lock()
        self.settings = {}
        self.worker_error = None
        self.last_exposure = None
        self.last_gain = None
        self.last_ae_enable = None
        self.fps = 0.0
        self.after_id = None
        self.kalman = Kalman2D()
        self.pid_x = PID(kP=0.0075, kI=0.025, kD=0.000005, output_bound_low=-12, output_bound_high=12)
        self.pid_y = PID(kP=0.01, kI=0.02, kD=0.000005, output_bound_low=-12, output_bound_high=12)
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        self.servo_deg_per_step = 0.24
        self.stab_pan_residual_deg = 0.0
        self.stab_tilt_residual_deg = 0.0
        self.stab_pan_filtered_deg = 0.0
        self.stab_tilt_filtered_deg = 0.0
        self.active_pan_id = 1
        self.active_tilt_id = 2
        self.jog_step_deg = tk.DoubleVar(value=1.0)
        self.servo_mode = tk.StringVar(value="无刷RS485")
        self.servo_status_mode = tk.StringVar(value=self.servo_mode.get())
        self.servo_status_pan = tk.StringVar(value="-")
        self.servo_status_tilt = tk.StringVar(value="-")
        self.servo_status_voltage = tk.StringVar(value="-")
        self.imu_status_pitch = tk.StringVar(value="-")
        self.imu_status_yaw = tk.StringVar(value="-")
        self.imu_status_age = tk.StringVar(value="-")
        self.imu_status_pitch_base = tk.StringVar(value="-")
        self.imu_status_yaw_base = tk.StringVar(value="-")
        self.imu_status_pitch_delta = tk.StringVar(value="-")
        self.imu_status_yaw_delta = tk.StringVar(value="-")
        self.latest_servo_status = None
        self.last_servo_status_time = 0.0
        self._board_transport_cls = None
        self._board_driver_cls = None
        self._bus_servo_cls = None
        self._brushless_motor_config_cls = None
        self._brushless_motor_cls = None

        self.port = tk.StringVar(value="/dev/ttyAMA1")
        # self.baudrate = tk.IntVar(value=115200)
        self.baudrate = tk.IntVar(value=1000000)
        self.brushless_pan_dev = tk.StringVar(value="/dev/ttySC0")
        self.brushless_tilt_dev = tk.StringVar(value="/dev/ttySC1")
        self.brushless_pan_baudrate = tk.IntVar(value=1000000)
        self.brushless_tilt_baudrate = tk.IntVar(value=1000000)
        self.brushless_pan_txden = tk.IntVar(value=22)
        self.brushless_tilt_txden = tk.IntVar(value=27)
        self.brushless_pan_direction_sign = tk.IntVar(value=1)
        self.brushless_tilt_direction_sign = tk.IntVar(value=1)
        self.brushless_pan_speed_dps = tk.DoubleVar(value=120.0)
        self.brushless_tilt_speed_dps = tk.DoubleVar(value=120.0)
        self.imu_port = tk.StringVar(value="/dev/ttyUSB0")
        self.imu_baudrate = tk.IntVar(value=9600)
        self.imu_use_6axis = tk.BooleanVar(value=True)
        self.imu_output_hz = tk.IntVar(value=50)
        self.imu_ax_offset_g = tk.DoubleVar(value=0.0)
        self.imu_ay_offset_g = tk.DoubleVar(value=0.0)
        self.imu_az_offset_g = tk.DoubleVar(value=0.0)
        self.imu_gx_offset_dps = tk.DoubleVar(value=0.0)
        self.imu_gy_offset_dps = tk.DoubleVar(value=0.0)
        self.imu_gz_offset_dps = tk.DoubleVar(value=0.0)
        self.imu_hx_offset = tk.IntVar(value=0)
        self.imu_hy_offset = tk.IntVar(value=0)
        self.imu_hz_offset = tk.IntVar(value=0)
        self.imu_az_reference_g = tk.DoubleVar(value=1.0)
        self.pan_id = tk.IntVar(value=1)
        self.tilt_id = tk.IntVar(value=2)
        self.move_time_ms = tk.IntVar(value=40)
        self.control_period_ms = tk.IntVar(value=50)
        self.track_enabled = tk.BooleanVar(value=True)
        self.pan_enabled = tk.BooleanVar(value=True)
        self.tilt_enabled = tk.BooleanVar(value=True)
        self.kp_x = tk.DoubleVar(value=0.0075)
        self.ki_x = tk.DoubleVar(value=0.025)
        self.kd_x = tk.DoubleVar(value=0.000005)
        self.kp_y = tk.DoubleVar(value=0.01)
        self.ki_y = tk.DoubleVar(value=0.02)
        self.kd_y = tk.DoubleVar(value=0.000005)
        self.error_deadband = tk.DoubleVar(value=3.0)
        self.max_delta_deg_per_sec = tk.DoubleVar(value=120.0)
        self.exposure_value = tk.IntVar(value=10000) # 微秒 us
        self.analogue_gain = tk.DoubleVar(value=1.0)
        self.ae_enable = tk.BooleanVar(value=True)
        self.ksize = tk.IntVar(value=5)
        self.min_dist = tk.IntVar(value=80)
        self.param1 = tk.IntVar(value=220)
        self.param2 = tk.IntVar(value=35)
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=120)
        self.x_bias = tk.IntVar(value=0)
        self.y_bias = tk.IntVar(value=0)
        self.camera_fps = tk.IntVar(value=60)
        self.image_rotate_deg = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="就绪")
        self.status_log_widget = None
        self.show_debug_panels = tk.BooleanVar(value=False)

        self.auto_stabilize = tk.BooleanVar(value=False)
        self.stab_gain_pitch = tk.DoubleVar(value=1.0)
        self.stab_gain_yaw = tk.DoubleVar(value=1.0)
        self.stab_pitch_deadband_deg = tk.DoubleVar(value=0.6)
        self.stab_tilt_limit_deg = tk.DoubleVar(value=8.0)
        self.stab_tilt_alpha = tk.DoubleVar(value=0.35)
        self.stab_tilt_rate_limit_deg_per_s = tk.DoubleVar(value=120.0)
        self.stab_yaw_deadband_deg = tk.DoubleVar(value=0.6)
        self.stab_pan_limit_deg = tk.DoubleVar(value=8.0)
        self.stab_pan_alpha = tk.DoubleVar(value=0.35)
        self.stab_pan_rate_limit_deg_per_s = tk.DoubleVar(value=120.0)
        
        # 激光指示对准配置
        self.laser_align_mode = tk.BooleanVar(value=False) # False:盲对准, True:指示对准
        self.laser_threshold = tk.IntVar(value=240)        # 激光二值化阈值
        
        # 舵机角度范围配置（角度制）
        self.pan_min = tk.DoubleVar(value=-180.0)
        self.pan_max = tk.DoubleVar(value=180.0)
        self.tilt_min = tk.DoubleVar(value=-90.0)
        self.tilt_max = tk.DoubleVar(value=90.0)
        
        # 卡尔曼滤波参数
        self.kalman_process_noise = tk.DoubleVar(value=0.03)
        self.kalman_measurement_noise = tk.DoubleVar(value=0.4)
        
        # 硬件物理边界（从舵机读取）
        self.hw_pan_min = tk.DoubleVar(value=-180.0)
        self.hw_pan_max = tk.DoubleVar(value=180.0)
        self.hw_tilt_min = tk.DoubleVar(value=-90.0)
        self.hw_tilt_max = tk.DoubleVar(value=90.0)
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_detection = None
        self.latest_detection_time = 0.0
        self.detect_stale_sec = 0.3
        self.latest_green_channel = None

        self.fps_cam = 0.0
        self.fps_ctrl = 0.0
        self.last_cam_frame_id = 0
        self.last_cam_time = 0.0
        
        # 激光测距查询控制
        self.last_laser_query_time = 0.0
        
        # 记录激光是否已经进入圆形标志物内部的状态
        self.laser_locked_in_circle = False

        # 用于平滑检测结果的EMA（指数移动平均）状态
        self.smoothed_detection = None
        self.ema_alpha = 0.3  # 平滑系数，越小越平滑但延迟越大，越大响应越快但抖动越大

        self._autosave_after_id = None
        self._autosave_suppress = True

        self._load_settings()
        self._build_ui()
        self.status_text.trace_add("write", self._on_status_text_changed)
        self.servo_mode.trace_add("write", self._on_servo_mode_change)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _e: self.on_close())
        self.root.bind("q", lambda _e: self.on_close())
        self._update_settings_from_vars()
        self._attach_autosave()
        self._autosave_suppress = False
        
        # 1. 强制配置激光测距模块为查询模式 (Passive / Inquire)
        configure_laser_module(
            port="/dev/ttyAMA3", 
            baudrate=115200, 
            module_id=0,
            output_mode="inquire",
            range_mode="medium",
            interface_mode="uart",
            uart_baudrate=115200
        )
        
        # 释放激光配置串口后，硬延时 1 秒，等待系统资源完全释放和稳定
        print("[INFO] Laser configured. Waiting 1.0s before initializing servos...")
        time.sleep(1.0)
        
        # 2. 尝试提前初始化舵机并在GUI显示前立即回正
        try:
            # 确保使用正确的配置初始化舵机
            fallback_mode = "无刷RS485"
            try:
                settings_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracker_settings.txt")
                if os.path.exists(settings_path):
                    with open(settings_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        fallback_mode = str(data.get("servo_mode", "无刷RS485")).strip()
            except Exception:
                pass
            
            # 在启动前设置模式，确保 _ensure_servo 使用正确的类
            self.servo_mode.set(fallback_mode)
            self._ensure_servo()

            try:
                self._ensure_imu()
                self._zero_imu()
            except Exception as exc:
                print(f"[WARNING] IMU init/zero failed before GUI start: {exc}")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[WARNING] Servo init failed before GUI start: {exc}")
            
        # 舵机回正指令发送后，硬延时 1 秒，等待舵机机械运动到位及总线电平恢复
        print("[INFO] Servos centered. Waiting 1.0s before attaching laser monitor...")
        time.sleep(1.0)
            
        # 3. 初始化激光测距模块 (被动查询模式)
        # 注意：这里不再调用 start() 开启后台死循环，而是由主循环按需调用 query_once()
        try:
            self.laser_ranger = LaserRangerQueryMonitor(port="/dev/ttyAMA3", baudrate=115200, module_id=0, history_len=10)
        except Exception as exc:
            print(f"[WARNING] Laser Ranger init failed: {exc}")
            
        self.root.after(10, self._start_runtime)

    def _update_settings_from_vars(self):
        fallback = {
            "servo_mode": "无刷RS485",
            "baudrate": 1000000,
            "brushless_pan_dev": "/dev/ttySC0",
            "brushless_tilt_dev": "/dev/ttySC1",
            "brushless_pan_baudrate": 1000000,
            "brushless_tilt_baudrate": 1000000,
            "brushless_pan_txden": 22,
            "brushless_tilt_txden": 27,
            "brushless_pan_direction_sign": 1,
            "brushless_tilt_direction_sign": 1,
            "brushless_pan_speed_dps": 120.0,
            "brushless_tilt_speed_dps": 120.0,
            "imu_port": "/dev/ttyUSB0",
            "imu_baudrate": 9600,
            "imu_use_6axis": True,
            "imu_output_hz": 50,
            "pan_id": 1,
            "tilt_id": 2,
            "move_time_ms": 40,
            "control_period_ms": 50,
            "kp_x": 0.0075,
            "ki_x": 0.025,
            "kd_x": 0.000005,
            "kp_y": 0.01,
            "ki_y": 0.02,
            "kd_y": 0.000005,
            "deadband": 3.0,
            "max_delta_deg_per_sec": 120.0,
            "exposure": 0.0,
            "gain": 8.0,
            "ksize": 5,
            "min_dist": 80,
            "param1": 220,
            "param2": 35,
            "min_radius": 20,
            "max_radius": 120,
            "x_bias": 0,
            "y_bias": 0,
            "camera_fps": 60,
            "image_rotate_deg": 0.0,
            "show_debug_panels": False,
            "laser_align_mode": False,
            "laser_threshold": 240,
            "pan_min": -180.0,
            "pan_max": 180.0,
            "tilt_min": -90.0,
            "tilt_max": 90.0,
            "hw_pan_min": -180.0,
            "hw_pan_max": 180.0,
            "hw_tilt_min": -90.0,
            "hw_tilt_max": 90.0,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.4,
            "auto_stabilize": False,
            "stab_gain_pitch": 1.0,
            "stab_gain_yaw": 1.0,
            "stab_pitch_deadband_deg": 0.6,
            "stab_tilt_limit_deg": 8.0,
            "stab_tilt_alpha": 0.35,
            "stab_tilt_rate_limit_deg_per_s": 120.0,
            "stab_yaw_deadband_deg": 0.6,
            "stab_pan_limit_deg": 8.0,
            "stab_pan_alpha": 0.35,
            "stab_pan_rate_limit_deg_per_s": 120.0,
            "imu_ax_offset_g": 0.0,
            "imu_ay_offset_g": 0.0,
            "imu_az_offset_g": 0.0,
            "imu_gx_offset_dps": 0.0,
            "imu_gy_offset_dps": 0.0,
            "imu_gz_offset_dps": 0.0,
            "imu_hx_offset": 0,
            "imu_hy_offset": 0,
            "imu_hz_offset": 0,
            "imu_az_reference_g": 1.0,
        }
        def safe_int(var, key):
            try:
                return int(var.get())
            except Exception:
                value = int(fallback[key])
                var.set(value)
                return value
        def safe_float(var, key):
            try:
                return float(var.get())
            except Exception:
                value = float(fallback[key])
                var.set(value)
                return value
        def safe_bool(var):
            try:
                return bool(var.get())
            except Exception:
                var.set(False)
                return False
        with self.settings_lock:
            self.settings = {
                "port": self.port.get(),
                "servo_mode": self.servo_mode.get(),
                "baudrate": safe_int(self.baudrate, "baudrate"),
                "brushless_pan_dev": str(self.brushless_pan_dev.get()),
                "brushless_tilt_dev": str(self.brushless_tilt_dev.get()),
                "brushless_pan_baudrate": safe_int(self.brushless_pan_baudrate, "brushless_pan_baudrate"),
                "brushless_tilt_baudrate": safe_int(self.brushless_tilt_baudrate, "brushless_tilt_baudrate"),
                "brushless_pan_txden": safe_int(self.brushless_pan_txden, "brushless_pan_txden"),
                "brushless_tilt_txden": safe_int(self.brushless_tilt_txden, "brushless_tilt_txden"),
                "brushless_pan_direction_sign": safe_int(self.brushless_pan_direction_sign, "brushless_pan_direction_sign"),
                "brushless_tilt_direction_sign": safe_int(self.brushless_tilt_direction_sign, "brushless_tilt_direction_sign"),
                "brushless_pan_speed_dps": safe_float(self.brushless_pan_speed_dps, "brushless_pan_speed_dps"),
                "brushless_tilt_speed_dps": safe_float(self.brushless_tilt_speed_dps, "brushless_tilt_speed_dps"),
                "imu_port": str(self.imu_port.get()),
                "imu_baudrate": safe_int(self.imu_baudrate, "imu_baudrate"),
                "imu_use_6axis": safe_bool(self.imu_use_6axis),
                "imu_output_hz": safe_int(self.imu_output_hz, "imu_output_hz"),
                "imu_ax_offset_g": safe_float(self.imu_ax_offset_g, "imu_ax_offset_g"),
                "imu_ay_offset_g": safe_float(self.imu_ay_offset_g, "imu_ay_offset_g"),
                "imu_az_offset_g": safe_float(self.imu_az_offset_g, "imu_az_offset_g"),
                "imu_gx_offset_dps": safe_float(self.imu_gx_offset_dps, "imu_gx_offset_dps"),
                "imu_gy_offset_dps": safe_float(self.imu_gy_offset_dps, "imu_gy_offset_dps"),
                "imu_gz_offset_dps": safe_float(self.imu_gz_offset_dps, "imu_gz_offset_dps"),
                "imu_hx_offset": safe_int(self.imu_hx_offset, "imu_hx_offset"),
                "imu_hy_offset": safe_int(self.imu_hy_offset, "imu_hy_offset"),
                "imu_hz_offset": safe_int(self.imu_hz_offset, "imu_hz_offset"),
                "imu_az_reference_g": safe_float(self.imu_az_reference_g, "imu_az_reference_g"),
                "pan_id": safe_int(self.pan_id, "pan_id"),
                "tilt_id": safe_int(self.tilt_id, "tilt_id"),
                "move_time_ms": safe_int(self.move_time_ms, "move_time_ms"),
                "control_period_ms": safe_int(self.control_period_ms, "control_period_ms"),
                "track_enabled": safe_bool(self.track_enabled),
                "pan_enabled": safe_bool(self.pan_enabled),
                "tilt_enabled": safe_bool(self.tilt_enabled),
                "kp_x": safe_float(self.kp_x, "kp_x"),
                "ki_x": safe_float(self.ki_x, "ki_x"),
                "kd_x": safe_float(self.kd_x, "kd_x"),
                "kp_y": safe_float(self.kp_y, "kp_y"),
                "ki_y": safe_float(self.ki_y, "ki_y"),
                "kd_y": safe_float(self.kd_y, "kd_y"),
                "deadband": safe_float(self.error_deadband, "deadband"),
                "max_delta_deg_per_sec": safe_float(self.max_delta_deg_per_sec, "max_delta_deg_per_sec"),
                "exposure": safe_float(self.exposure_value, "exposure"),
                "gain": safe_float(self.analogue_gain, "gain"),
                "ae_enable": safe_bool(self.ae_enable),
                "ksize": safe_int(self.ksize, "ksize"),
                "min_dist": safe_int(self.min_dist, "min_dist"),
                "param1": safe_int(self.param1, "param1"),
                "param2": safe_int(self.param2, "param2"),
                "min_radius": safe_int(self.min_radius, "min_radius"),
                "max_radius": safe_int(self.max_radius, "max_radius"),
                "x_bias": safe_int(self.x_bias, "x_bias"),
                "y_bias": safe_int(self.y_bias, "y_bias"),
                "camera_fps": safe_int(self.camera_fps, "camera_fps"),
                "image_rotate_deg": safe_float(self.image_rotate_deg, "image_rotate_deg"),
                "show_debug_panels": safe_bool(self.show_debug_panels),
                "laser_align_mode": safe_bool(self.laser_align_mode),
                "laser_threshold": safe_int(self.laser_threshold, "laser_threshold"),
                "pan_min": safe_float(self.pan_min, "pan_min"),
                "pan_max": safe_float(self.pan_max, "pan_max"),
            "tilt_min": safe_float(self.tilt_min, "tilt_min"),
            "tilt_max": safe_float(self.tilt_max, "tilt_max"),
            "kalman_process_noise": safe_float(self.kalman_process_noise, "kalman_process_noise"),
            "kalman_measurement_noise": safe_float(self.kalman_measurement_noise, "kalman_measurement_noise"),
                "auto_stabilize": safe_bool(self.auto_stabilize),
                "stab_gain_pitch": safe_float(self.stab_gain_pitch, "stab_gain_pitch"),
                "stab_gain_yaw": safe_float(self.stab_gain_yaw, "stab_gain_yaw"),
                "stab_pitch_deadband_deg": safe_float(self.stab_pitch_deadband_deg, "stab_pitch_deadband_deg"),
                "stab_tilt_limit_deg": safe_float(self.stab_tilt_limit_deg, "stab_tilt_limit_deg"),
                "stab_tilt_alpha": safe_float(self.stab_tilt_alpha, "stab_tilt_alpha"),
                "stab_tilt_rate_limit_deg_per_s": safe_float(self.stab_tilt_rate_limit_deg_per_s, "stab_tilt_rate_limit_deg_per_s"),
                "stab_yaw_deadband_deg": safe_float(self.stab_yaw_deadband_deg, "stab_yaw_deadband_deg"),
                "stab_pan_limit_deg": safe_float(self.stab_pan_limit_deg, "stab_pan_limit_deg"),
                "stab_pan_alpha": safe_float(self.stab_pan_alpha, "stab_pan_alpha"),
                "stab_pan_rate_limit_deg_per_s": safe_float(self.stab_pan_rate_limit_deg_per_s, "stab_pan_rate_limit_deg_per_s"),
        }

    def _get_settings(self):
        # 实时从GUI变量读取，确保修改立即生效
        defaults = {
            "servo_mode": "无刷RS485",
            "port": "/dev/ttyAMA1",
            "baudrate": 1000000,
            "brushless_pan_dev": "/dev/ttySC0",
            "brushless_tilt_dev": "/dev/ttySC1",
            "brushless_pan_baudrate": 1000000,
            "brushless_tilt_baudrate": 1000000,
            "brushless_pan_txden": 22,
            "brushless_tilt_txden": 27,
            "brushless_pan_direction_sign": 1,
            "brushless_tilt_direction_sign": 1,
            "brushless_pan_speed_dps": 120.0,
            "brushless_tilt_speed_dps": 120.0,
            "imu_port": "/dev/ttyUSB0",
            "imu_baudrate": 9600,
            "imu_use_6axis": True,
            "imu_output_hz": 50,
            "imu_output_hz": 50,
            "imu_ax_offset_g": 0.0,
            "imu_ay_offset_g": 0.0,
            "imu_az_offset_g": 0.0,
            "imu_gx_offset_dps": 0.0,
            "imu_gy_offset_dps": 0.0,
            "imu_gz_offset_dps": 0.0,
            "imu_hx_offset": 0,
            "imu_hy_offset": 0,
            "imu_hz_offset": 0,
            "imu_az_reference_g": 1.0,
            "pan_id": 1,
            "tilt_id": 2,
            "move_time_ms": 40,
            "control_period_ms": 50,
            "track_enabled": True,
            "pan_enabled": True,
            "tilt_enabled": True,
            "kp_x": 0.0075,
            "ki_x": 0.025,
            "kd_x": 0.000005,
            "kp_y": 0.0075,
            "ki_y": 0.025,
            "kd_y": 0.000005,
            "deadband": 3.0,
            "max_delta_deg_per_sec": 120.0,
            "exposure": 10000,
            "gain": 1.0,
            "ae_enable": True,
            "ksize": 5,
            "min_dist": 80,
            "param1": 220,
            "param2": 35,
            "min_radius": 20,
            "max_radius": 120,
            "x_bias": 0,
            "y_bias": 0,
            "image_rotate_deg": 0.0,
            "show_debug_panels": False,
            "pan_min": -180.0,
            "pan_max": 180.0,
            "tilt_min": -90.0,
            "tilt_max": 90.0,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.4,
            "auto_stabilize": False,
            "stab_gain_pitch": 1.0,
            "stab_gain_yaw": 1.0,
            "stab_pitch_deadband_deg": 0.6,
            "stab_tilt_limit_deg": 8.0,
            "stab_tilt_alpha": 0.35,
            "stab_tilt_rate_limit_deg_per_s": 120.0,
            "stab_yaw_deadband_deg": 0.6,
            "stab_pan_limit_deg": 8.0,
            "stab_pan_alpha": 0.35,
            "stab_pan_rate_limit_deg_per_s": 120.0,
        }
        def safe_int(var, key):
            try:
                v = var.get()
                return int(v) if v != "" else defaults[key]
            except Exception:
                return defaults[key]
        def safe_float(var, key):
            try:
                v = var.get()
                return float(v) if v != "" else defaults[key]
            except Exception:
                return defaults[key]
        def safe_bool(var, key):
            try:
                return bool(var.get())
            except Exception:
                return defaults[key]
        return {
            "servo_mode": str(self.servo_mode.get()) if self.servo_mode.get() else defaults["servo_mode"],
            "port": str(self.port.get()) if self.port.get() else defaults["port"],
            "baudrate": safe_int(self.baudrate, "baudrate"),
            "brushless_pan_dev": str(self.brushless_pan_dev.get()) if self.brushless_pan_dev.get() else defaults["brushless_pan_dev"],
            "brushless_tilt_dev": str(self.brushless_tilt_dev.get()) if self.brushless_tilt_dev.get() else defaults["brushless_tilt_dev"],
            "brushless_pan_baudrate": safe_int(self.brushless_pan_baudrate, "brushless_pan_baudrate"),
            "brushless_tilt_baudrate": safe_int(self.brushless_tilt_baudrate, "brushless_tilt_baudrate"),
            "brushless_pan_txden": safe_int(self.brushless_pan_txden, "brushless_pan_txden"),
            "brushless_tilt_txden": safe_int(self.brushless_tilt_txden, "brushless_tilt_txden"),
            "brushless_pan_direction_sign": safe_int(self.brushless_pan_direction_sign, "brushless_pan_direction_sign"),
            "brushless_tilt_direction_sign": safe_int(self.brushless_tilt_direction_sign, "brushless_tilt_direction_sign"),
            "brushless_pan_speed_dps": safe_float(self.brushless_pan_speed_dps, "brushless_pan_speed_dps"),
            "brushless_tilt_speed_dps": safe_float(self.brushless_tilt_speed_dps, "brushless_tilt_speed_dps"),
            "imu_port": str(self.imu_port.get()) if self.imu_port.get() else defaults["imu_port"],
            "imu_baudrate": safe_int(self.imu_baudrate, "imu_baudrate"),
            "imu_use_6axis": safe_bool(self.imu_use_6axis, "imu_use_6axis"),
            "imu_output_hz": safe_int(self.imu_output_hz, "imu_output_hz"),
            "imu_ax_offset_g": safe_float(self.imu_ax_offset_g, "imu_ax_offset_g"),
            "imu_ay_offset_g": safe_float(self.imu_ay_offset_g, "imu_ay_offset_g"),
            "imu_az_offset_g": safe_float(self.imu_az_offset_g, "imu_az_offset_g"),
            "imu_gx_offset_dps": safe_float(self.imu_gx_offset_dps, "imu_gx_offset_dps"),
            "imu_gy_offset_dps": safe_float(self.imu_gy_offset_dps, "imu_gy_offset_dps"),
            "imu_gz_offset_dps": safe_float(self.imu_gz_offset_dps, "imu_gz_offset_dps"),
            "imu_hx_offset": safe_int(self.imu_hx_offset, "imu_hx_offset"),
            "imu_hy_offset": safe_int(self.imu_hy_offset, "imu_hy_offset"),
            "imu_hz_offset": safe_int(self.imu_hz_offset, "imu_hz_offset"),
            "imu_az_reference_g": safe_float(self.imu_az_reference_g, "imu_az_reference_g"),
            "pan_id": safe_int(self.pan_id, "pan_id"),
            "tilt_id": safe_int(self.tilt_id, "tilt_id"),
            "move_time_ms": safe_int(self.move_time_ms, "move_time_ms"),
            "control_period_ms": safe_int(self.control_period_ms, "control_period_ms"),
            "track_enabled": safe_bool(self.track_enabled, "track_enabled"),
            "pan_enabled": safe_bool(self.pan_enabled, "pan_enabled"),
            "tilt_enabled": safe_bool(self.tilt_enabled, "tilt_enabled"),
            "kp_x": safe_float(self.kp_x, "kp_x"),
            "ki_x": safe_float(self.ki_x, "ki_x"),
            "kd_x": safe_float(self.kd_x, "kd_x"),
            "kp_y": safe_float(self.kp_y, "kp_y"),
            "ki_y": safe_float(self.ki_y, "ki_y"),
            "kd_y": safe_float(self.kd_y, "kd_y"),
            "deadband": safe_float(self.error_deadband, "deadband"),
            "max_delta_deg_per_sec": safe_float(self.max_delta_deg_per_sec, "max_delta_deg_per_sec"),
            "exposure": safe_int(self.exposure_value, "exposure"),
            "gain": safe_float(self.analogue_gain, "gain"),
            "ae_enable": safe_bool(self.ae_enable, "ae_enable"),
            "ksize": safe_int(self.ksize, "ksize"),
            "min_dist": safe_int(self.min_dist, "min_dist"),
            "param1": safe_int(self.param1, "param1"),
            "param2": safe_int(self.param2, "param2"),
            "min_radius": safe_int(self.min_radius, "min_radius"),
            "max_radius": safe_int(self.max_radius, "max_radius"),
            "x_bias": safe_int(self.x_bias, "x_bias"),
            "y_bias": safe_int(self.y_bias, "y_bias"),
            "camera_fps": safe_int(self.camera_fps, "camera_fps"),
            "image_rotate_deg": safe_float(self.image_rotate_deg, "image_rotate_deg"),
            "show_debug_panels": safe_bool(self.show_debug_panels, "show_debug_panels"),
            "laser_align_mode": safe_bool(self.laser_align_mode, "laser_align_mode"),
            "laser_threshold": safe_int(self.laser_threshold, "laser_threshold"),
            "pan_min": safe_float(self.pan_min, "pan_min"),
            "pan_max": safe_float(self.pan_max, "pan_max"),
            "tilt_min": safe_float(self.tilt_min, "tilt_min"),
            "tilt_max": safe_float(self.tilt_max, "tilt_max"),
            "kalman_process_noise": safe_float(self.kalman_process_noise, "kalman_process_noise"),
            "kalman_measurement_noise": safe_float(self.kalman_measurement_noise, "kalman_measurement_noise"),
            "auto_stabilize": safe_bool(self.auto_stabilize, "auto_stabilize"),
            "stab_gain_pitch": safe_float(self.stab_gain_pitch, "stab_gain_pitch"),
            "stab_gain_yaw": safe_float(self.stab_gain_yaw, "stab_gain_yaw"),
            "stab_pitch_deadband_deg": safe_float(self.stab_pitch_deadband_deg, "stab_pitch_deadband_deg"),
            "stab_tilt_limit_deg": safe_float(self.stab_tilt_limit_deg, "stab_tilt_limit_deg"),
            "stab_tilt_alpha": safe_float(self.stab_tilt_alpha, "stab_tilt_alpha"),
            "stab_tilt_rate_limit_deg_per_s": safe_float(self.stab_tilt_rate_limit_deg_per_s, "stab_tilt_rate_limit_deg_per_s"),
            "stab_yaw_deadband_deg": safe_float(self.stab_yaw_deadband_deg, "stab_yaw_deadband_deg"),
            "stab_pan_limit_deg": safe_float(self.stab_pan_limit_deg, "stab_pan_limit_deg"),
            "stab_pan_alpha": safe_float(self.stab_pan_alpha, "stab_pan_alpha"),
            "stab_pan_rate_limit_deg_per_s": safe_float(self.stab_pan_rate_limit_deg_per_s, "stab_pan_rate_limit_deg_per_s"),
            "hw_pan_min": safe_float(self.hw_pan_min, "hw_pan_min"),
            "hw_pan_max": safe_float(self.hw_pan_max, "hw_pan_max"),
            "hw_tilt_min": safe_float(self.hw_tilt_min, "hw_tilt_min"),
            "hw_tilt_max": safe_float(self.hw_tilt_max, "hw_tilt_max"),
        }

    def _settings_path(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(base_dir, "tracker_settings.txt")

    def _load_settings(self):
        path = self._settings_path()
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return
        var_map = {
            "servo_mode": self.servo_mode,
            "port": self.port,
            "baudrate": self.baudrate,
            "brushless_pan_dev": self.brushless_pan_dev,
            "brushless_tilt_dev": self.brushless_tilt_dev,
            "brushless_pan_baudrate": self.brushless_pan_baudrate,
            "brushless_tilt_baudrate": self.brushless_tilt_baudrate,
            "brushless_pan_txden": self.brushless_pan_txden,
            "brushless_tilt_txden": self.brushless_tilt_txden,
            "brushless_pan_direction_sign": self.brushless_pan_direction_sign,
            "brushless_tilt_direction_sign": self.brushless_tilt_direction_sign,
            "brushless_pan_speed_dps": self.brushless_pan_speed_dps,
            "brushless_tilt_speed_dps": self.brushless_tilt_speed_dps,
            "imu_port": self.imu_port,
            "imu_baudrate": self.imu_baudrate,
            "imu_use_6axis": self.imu_use_6axis,
            "imu_output_hz": self.imu_output_hz,
            "imu_ax_offset_g": self.imu_ax_offset_g,
            "imu_ay_offset_g": self.imu_ay_offset_g,
            "imu_az_offset_g": self.imu_az_offset_g,
            "imu_gx_offset_dps": self.imu_gx_offset_dps,
            "imu_gy_offset_dps": self.imu_gy_offset_dps,
            "imu_gz_offset_dps": self.imu_gz_offset_dps,
            "imu_hx_offset": self.imu_hx_offset,
            "imu_hy_offset": self.imu_hy_offset,
            "imu_hz_offset": self.imu_hz_offset,
            "imu_az_reference_g": self.imu_az_reference_g,
            "pan_id": self.pan_id,
            "tilt_id": self.tilt_id,
            "move_time_ms": self.move_time_ms,
            "control_period_ms": self.control_period_ms,
            "track_enabled": self.track_enabled,
            "pan_enabled": self.pan_enabled,
            "tilt_enabled": self.tilt_enabled,
            "kp_x": self.kp_x,
            "ki_x": self.ki_x,
            "kd_x": self.kd_x,
            "kp_y": self.kp_y,
            "ki_y": self.ki_y,
            "kd_y": self.kd_y,
            "deadband": self.error_deadband,
            "max_delta_deg_per_sec": self.max_delta_deg_per_sec,
            "exposure": self.exposure_value,
            "gain": self.analogue_gain,
            "ae_enable": self.ae_enable,
            "ksize": self.ksize,
            "min_dist": self.min_dist,
            "param1": self.param1,
            "param2": self.param2,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "x_bias": self.x_bias,
            "y_bias": self.y_bias,
            "image_rotate_deg": self.image_rotate_deg,
            "show_debug_panels": self.show_debug_panels,
            "pan_min": self.pan_min,
            "pan_max": self.pan_max,
            "tilt_min": self.tilt_min,
            "tilt_max": self.tilt_max,
            "kalman_process_noise": self.kalman_process_noise,
            "kalman_measurement_noise": self.kalman_measurement_noise,
            "auto_stabilize": self.auto_stabilize,
            "stab_gain_pitch": self.stab_gain_pitch,
            "stab_gain_yaw": self.stab_gain_yaw,
            "stab_pitch_deadband_deg": self.stab_pitch_deadband_deg,
            "stab_tilt_limit_deg": self.stab_tilt_limit_deg,
            "stab_tilt_alpha": self.stab_tilt_alpha,
            "stab_tilt_rate_limit_deg_per_s": self.stab_tilt_rate_limit_deg_per_s,
            "stab_yaw_deadband_deg": self.stab_yaw_deadband_deg,
            "stab_pan_limit_deg": self.stab_pan_limit_deg,
            "stab_pan_alpha": self.stab_pan_alpha,
            "stab_pan_rate_limit_deg_per_s": self.stab_pan_rate_limit_deg_per_s,
        }
        for key, var in var_map.items():
            if key not in data:
                continue
            value = data[key]
            if value is None or value == "":
                continue
            if isinstance(var, tk.BooleanVar):
                var.set(bool(value))
            elif isinstance(var, tk.IntVar):
                var.set(int(value))
            elif isinstance(var, tk.DoubleVar):
                var.set(float(value))
            else:
                if key == "servo_mode":
                    var.set(str(value).strip())
                else:
                    var.set(value)

    def _save_settings(self, quiet=False):
        path = self._settings_path()
        try:
            data = {
                "servo_mode": self.servo_mode.get(),
                "port": self.port.get(),
                "baudrate": int(self.baudrate.get()),
                "brushless_pan_dev": self.brushless_pan_dev.get(),
                "brushless_tilt_dev": self.brushless_tilt_dev.get(),
                "brushless_pan_baudrate": int(self.brushless_pan_baudrate.get()),
                "brushless_tilt_baudrate": int(self.brushless_tilt_baudrate.get()),
                "brushless_pan_txden": int(self.brushless_pan_txden.get()),
                "brushless_tilt_txden": int(self.brushless_tilt_txden.get()),
                "brushless_pan_direction_sign": int(self.brushless_pan_direction_sign.get()),
                "brushless_tilt_direction_sign": int(self.brushless_tilt_direction_sign.get()),
                "brushless_pan_speed_dps": float(self.brushless_pan_speed_dps.get()),
                "brushless_tilt_speed_dps": float(self.brushless_tilt_speed_dps.get()),
                "imu_port": self.imu_port.get(),
                "imu_baudrate": int(self.imu_baudrate.get()),
                "imu_use_6axis": bool(self.imu_use_6axis.get()),
                "imu_output_hz": int(self.imu_output_hz.get()),
                "imu_ax_offset_g": float(self.imu_ax_offset_g.get()),
                "imu_ay_offset_g": float(self.imu_ay_offset_g.get()),
                "imu_az_offset_g": float(self.imu_az_offset_g.get()),
                "imu_gx_offset_dps": float(self.imu_gx_offset_dps.get()),
                "imu_gy_offset_dps": float(self.imu_gy_offset_dps.get()),
                "imu_gz_offset_dps": float(self.imu_gz_offset_dps.get()),
                "imu_hx_offset": int(self.imu_hx_offset.get()),
                "imu_hy_offset": int(self.imu_hy_offset.get()),
                "imu_hz_offset": int(self.imu_hz_offset.get()),
                "imu_az_reference_g": float(self.imu_az_reference_g.get()),
                "pan_id": int(self.pan_id.get()),
                "tilt_id": int(self.tilt_id.get()),
                "move_time_ms": int(self.move_time_ms.get()),
                "control_period_ms": int(self.control_period_ms.get()),
                "track_enabled": bool(self.track_enabled.get()),
                "pan_enabled": bool(self.pan_enabled.get()),
                "tilt_enabled": bool(self.tilt_enabled.get()),
                "kp_x": float(self.kp_x.get()),
                "ki_x": float(self.ki_x.get()),
                "kd_x": float(self.kd_x.get()),
                "kp_y": float(self.kp_y.get()),
                "ki_y": float(self.ki_y.get()),
                "kd_y": float(self.kd_y.get()),
                "deadband": float(self.error_deadband.get()),
                "max_delta_deg_per_sec": float(self.max_delta_deg_per_sec.get()),
                "exposure": float(self.exposure_value.get()),
                "gain": float(self.analogue_gain.get()),
                "ae_enable": bool(self.ae_enable.get()),
                "ksize": int(self.ksize.get()),
                "min_dist": int(self.min_dist.get()),
                "param1": int(self.param1.get()),
                "param2": int(self.param2.get()),
                "min_radius": int(self.min_radius.get()),
                "max_radius": int(self.max_radius.get()),
                "x_bias": int(self.x_bias.get()),
                "y_bias": int(self.y_bias.get()),
                "camera_fps": int(self.camera_fps.get()),
                "image_rotate_deg": float(self.image_rotate_deg.get()),
                "show_debug_panels": bool(self.show_debug_panels.get()),
                "laser_align_mode": bool(self.laser_align_mode.get()),
                "laser_threshold": int(self.laser_threshold.get()),
                "pan_min": float(self.pan_min.get()),
                "pan_max": float(self.pan_max.get()),
                "tilt_min": float(self.tilt_min.get()),
                "tilt_max": float(self.tilt_max.get()),
                "kalman_process_noise": float(self.kalman_process_noise.get()),
                "kalman_measurement_noise": float(self.kalman_measurement_noise.get()),
                "auto_stabilize": bool(self.auto_stabilize.get()),
                "stab_gain_pitch": float(self.stab_gain_pitch.get()),
                "stab_gain_yaw": float(self.stab_gain_yaw.get()),
                "stab_pitch_deadband_deg": float(self.stab_pitch_deadband_deg.get()),
                "stab_tilt_limit_deg": float(self.stab_tilt_limit_deg.get()),
                "stab_tilt_alpha": float(self.stab_tilt_alpha.get()),
                "stab_tilt_rate_limit_deg_per_s": float(self.stab_tilt_rate_limit_deg_per_s.get()),
                "stab_yaw_deadband_deg": float(self.stab_yaw_deadband_deg.get()),
                "stab_pan_limit_deg": float(self.stab_pan_limit_deg.get()),
                "stab_pan_alpha": float(self.stab_pan_alpha.get()),
                "stab_pan_rate_limit_deg_per_s": float(self.stab_pan_rate_limit_deg_per_s.get()),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            if not quiet:
                print(f"[INFO] 设置已保存到: {path}")
        except Exception as e:
            print(f"[ERROR] 保存设置失败: {e}")

    def _schedule_autosave(self):
        if self._autosave_suppress:
            return
        if self._autosave_after_id is not None:
            try:
                self.root.after_cancel(self._autosave_after_id)
            except Exception:
                pass
        self._autosave_after_id = self.root.after(600, self._run_autosave)

    def _run_autosave(self):
        self._autosave_after_id = None
        self._save_settings(quiet=True)

    def _attach_autosave(self):
        autosave_vars = [
            self.servo_mode,
            self.port,
            self.baudrate,
            self.brushless_pan_dev,
            self.brushless_tilt_dev,
            self.brushless_pan_baudrate,
            self.brushless_tilt_baudrate,
            self.brushless_pan_txden,
            self.brushless_tilt_txden,
            self.brushless_pan_direction_sign,
            self.brushless_tilt_direction_sign,
            self.brushless_pan_speed_dps,
            self.brushless_tilt_speed_dps,
            self.imu_port,
            self.imu_baudrate,
            self.imu_use_6axis,
            self.imu_output_hz,
            self.imu_ax_offset_g,
            self.imu_ay_offset_g,
            self.imu_az_offset_g,
            self.imu_gx_offset_dps,
            self.imu_gy_offset_dps,
            self.imu_gz_offset_dps,
            self.imu_hx_offset,
            self.imu_hy_offset,
            self.imu_hz_offset,
            self.imu_az_reference_g,
            self.pan_id,
            self.tilt_id,
            self.move_time_ms,
            self.control_period_ms,
            self.track_enabled,
            self.pan_enabled,
            self.tilt_enabled,
            self.kp_x,
            self.ki_x,
            self.kd_x,
            self.kp_y,
            self.ki_y,
            self.kd_y,
            self.error_deadband,
            self.max_delta_deg_per_sec,
            self.exposure_value,
            self.analogue_gain,
            self.ae_enable,
            self.ksize,
            self.min_dist,
            self.param1,
            self.param2,
            self.min_radius,
            self.max_radius,
            self.x_bias,
            self.y_bias,
            self.image_rotate_deg,
            self.show_debug_panels,
            self.camera_fps,
            self.laser_align_mode,
            self.laser_threshold,
            self.pan_min,
            self.pan_max,
            self.tilt_min,
            self.tilt_max,
            self.kalman_process_noise,
            self.kalman_measurement_noise,
            self.auto_stabilize,
            self.stab_gain_pitch,
            self.stab_gain_yaw,
            self.stab_pitch_deadband_deg,
            self.stab_tilt_limit_deg,
            self.stab_tilt_alpha,
            self.stab_tilt_rate_limit_deg_per_s,
            self.stab_yaw_deadband_deg,
            self.stab_pan_limit_deg,
            self.stab_pan_alpha,
            self.stab_pan_rate_limit_deg_per_s,
            self.hw_pan_min,
            self.hw_pan_max,
            self.hw_tilt_min,
            self.hw_tilt_max,
        ]
        for var in autosave_vars:
            try:
                var.trace_add("write", lambda *_args: self._schedule_autosave())
            except Exception:
                pass

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=420)
        left.pack(side=tk.LEFT, fill=tk.Y)
        # 移除 pack_propagate(False)，让左侧面板高度能根据内容和主窗口自适应，同时利用 width=420 作为建议宽度
        # left.pack_propagate(False)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right, textvariable=self.status_text).pack(anchor=tk.W, pady=(6, 0))
        ttk.Checkbutton(right, text="显示四宫格调试(更耗性能)", variable=self.show_debug_panels).pack(anchor=tk.W, pady=(2, 0))
        self.status_log_widget = scrolledtext.ScrolledText(right, height=6, wrap=tk.WORD)
        self.status_log_widget.pack(fill=tk.X, pady=(4, 0))
        self.status_log_widget.configure(state=tk.DISABLED)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_basic = ttk.Frame(notebook, padding=8)
        tab_pid = ttk.Frame(notebook, padding=8)
        tab_vision = ttk.Frame(notebook, padding=8)
        tab_camera = ttk.Frame(notebook, padding=8)
        notebook.add(tab_basic, text="基本")
        notebook.add(tab_pid, text="PID")
        notebook.add(tab_vision, text="视觉")
        notebook.add(tab_camera, text="相机")

        tab_basic.columnconfigure(1, weight=1)
        tab_basic.columnconfigure(3, weight=1)
        r = 0
        self.port_label = ttk.Label(tab_basic, text="串口")
        self.port_label.grid(row=r, column=0, sticky="w", padx=(0, 6), pady=(2, 2))
        self.port_entry = ttk.Entry(tab_basic, textvariable=self.port, width=18)
        self.port_entry.grid(row=r, column=1, sticky="ew", pady=(2, 2))
        self._grid_entry(tab_basic, r, 2, "移动时间ms", self.move_time_ms, width=8)
        r += 1
        self.baudrate_label = ttk.Label(tab_basic, text="波特率")
        self.baudrate_label.grid(row=r, column=0, sticky="w", padx=(0, 6), pady=(2, 2))
        self.baudrate_entry = ttk.Entry(tab_basic, textvariable=self.baudrate, width=10)
        self.baudrate_entry.grid(row=r, column=1, sticky="ew", pady=(2, 2))
        ttk.Label(tab_basic, text="控制方式").grid(row=r, column=2, sticky="w", padx=(0, 6), pady=(2, 2))
        mode_combo = ttk.Combobox(tab_basic, textvariable=self.servo_mode, values=("调试板", "控制板", "无刷RS485"), state="readonly", width=8)
        mode_combo.grid(row=r, column=3, sticky="w", pady=(2, 2))
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平ID", self.pan_id, width=8)
        self._grid_entry(tab_basic, r, 2, "俯仰ID", self.tilt_id, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平串口", self.brushless_pan_dev, width=12)
        self._grid_entry(tab_basic, r, 2, "俯仰串口", self.brushless_tilt_dev, width=12)
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平波特率", self.brushless_pan_baudrate, width=10)
        self._grid_entry(tab_basic, r, 2, "俯仰波特率", self.brushless_tilt_baudrate, width=10)
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平TXDEN", self.brushless_pan_txden, width=8)
        self._grid_entry(tab_basic, r, 2, "俯仰TXDEN", self.brushless_tilt_txden, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平方向", self.brushless_pan_direction_sign, width=8)
        self._grid_entry(tab_basic, r, 2, "俯仰方向", self.brushless_tilt_direction_sign, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平速度dps", self.brushless_pan_speed_dps, width=8)
        self._grid_entry(tab_basic, r, 2, "俯仰速度dps", self.brushless_tilt_speed_dps, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "控制周期ms", self.control_period_ms, width=8)
        self._grid_entry(tab_basic, r, 2, "点动角度", self.jog_step_deg, width=8)
        r += 1
        self._refresh_servo_mode_ui()
        ttk.Checkbutton(tab_basic, text="启用跟踪", variable=self.track_enabled).grid(row=r, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="启用水平", variable=self.pan_enabled).grid(row=r, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="启用俯仰", variable=self.tilt_enabled).grid(row=r, column=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="自动稳定", variable=self.auto_stabilize).grid(row=r, column=3, sticky="w", pady=(6, 0))
        r += 1
        btns = ttk.Frame(tab_basic)
        btns.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)
        btns.columnconfigure(3, weight=1)
        ttk.Button(btns, text="开始", command=self.start).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="停止", command=self.stop).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(btns, text="复位", command=self.reset_axes).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        ttk.Button(btns, text="退出", command=self.on_close).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        jog = ttk.LabelFrame(tab_basic, text="点动", padding=8)
        jog.grid(row=r + 1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        for c in range(3):
            jog.columnconfigure(c, weight=1)
        ttk.Button(jog, text="上", command=lambda: self._jog(0.0, +self.jog_step_deg.get())).grid(row=0, column=1, sticky="ew")
        ttk.Button(jog, text="左", command=lambda: self._jog(+self.jog_step_deg.get(), 0.0)).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(jog, text="下", command=lambda: self._jog(0.0, -self.jog_step_deg.get())).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(jog, text="右", command=lambda: self._jog(-self.jog_step_deg.get(), 0.0)).grid(row=1, column=2, sticky="ew", pady=(6, 0))

        status_frame = ttk.LabelFrame(tab_basic, text="舵机状态", padding=8)
        status_frame.grid(row=r + 2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        status_frame.columnconfigure(1, weight=1)
        status_frame.columnconfigure(3, weight=1)
        ttk.Label(status_frame, text="模式:").grid(row=0, column=0, sticky="e")
        ttk.Label(status_frame, textvariable=self.servo_status_mode).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(status_frame, text="水平:").grid(row=0, column=2, sticky="e")
        ttk.Label(status_frame, textvariable=self.servo_status_pan).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(status_frame, text="俯仰:").grid(row=1, column=0, sticky="e")
        ttk.Label(status_frame, textvariable=self.servo_status_tilt).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(status_frame, text="电压:").grid(row=1, column=2, sticky="e")
        ttk.Label(status_frame, textvariable=self.servo_status_voltage).grid(row=1, column=3, sticky="w", padx=5)

        imu_frame = ttk.LabelFrame(tab_basic, text="IMU稳定", padding=8)
        imu_frame.grid(row=r + 3, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        imu_frame.columnconfigure(1, weight=1)
        imu_frame.columnconfigure(3, weight=1)
        ttk.Label(imu_frame, text="IMU串口:").grid(row=0, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.imu_port, width=14).grid(row=0, column=1, sticky="ew", padx=5)
        ttk.Label(imu_frame, text="波特率:").grid(row=0, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.imu_baudrate, width=10).grid(row=0, column=3, sticky="ew", padx=5)
        ttk.Label(imu_frame, text="输出Hz:").grid(row=1, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.imu_output_hz, width=10).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Button(imu_frame, text="应用输出", command=self._apply_imu_output_rate).grid(row=1, column=2, sticky="ew", padx=(0, 5))
        ttk.Button(imu_frame, text="应用波特率", command=self._apply_imu_baudrate).grid(row=1, column=3, sticky="ew")
        ttk.Checkbutton(imu_frame, text="6轴算法", variable=self.imu_use_6axis).grid(row=2, column=0, sticky="w")
        ttk.Button(imu_frame, text="应用算法", command=self._apply_imu_algorithm_mode).grid(row=2, column=1, sticky="ew", padx=(5, 0))
        ttk.Label(imu_frame, text="Pitch增益:").grid(row=3, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_gain_pitch, width=10).grid(row=3, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw增益:").grid(row=3, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_gain_yaw, width=10).grid(row=3, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch死区(°):").grid(row=4, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pitch_deadband_deg, width=10).grid(row=4, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw死区(°):").grid(row=4, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_yaw_deadband_deg, width=10).grid(row=4, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch限幅(°):").grid(row=5, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_tilt_limit_deg, width=10).grid(row=5, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw限幅(°):").grid(row=5, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pan_limit_deg, width=10).grid(row=5, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch平滑α:").grid(row=6, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_tilt_alpha, width=10).grid(row=6, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw平滑α:").grid(row=6, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pan_alpha, width=10).grid(row=6, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch速度限幅(°/s):").grid(row=7, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_tilt_rate_limit_deg_per_s, width=10).grid(row=7, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw速度限幅(°/s):").grid(row=7, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pan_rate_limit_deg_per_s, width=10).grid(row=7, column=3, sticky="w", padx=5)
        ttk.Button(imu_frame, text="IMU置零", command=self._zero_imu).grid(row=8, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(imu_frame, text="零偏设置", command=self._open_imu_offsets_dialog).grid(row=8, column=1, sticky="ew", pady=(8, 0), padx=(5, 0))
        ttk.Label(imu_frame, text="Pitch:").grid(row=8, column=2, sticky="e", pady=(8, 0))
        ttk.Label(imu_frame, textvariable=self.imu_status_pitch).grid(row=8, column=3, sticky="w", pady=(8, 0))
        ttk.Label(imu_frame, text="Yaw:").grid(row=9, column=2, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_yaw).grid(row=9, column=3, sticky="w")
        ttk.Label(imu_frame, text="Age:").grid(row=9, column=0, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_age).grid(row=9, column=1, sticky="w")
        ttk.Label(imu_frame, text="基准Pitch:").grid(row=10, column=0, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_pitch_base).grid(row=10, column=1, sticky="w")
        ttk.Label(imu_frame, text="基准Yaw:").grid(row=10, column=2, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_yaw_base).grid(row=10, column=3, sticky="w")
        ttk.Label(imu_frame, text="ΔPitch:").grid(row=11, column=0, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_pitch_delta).grid(row=11, column=1, sticky="w")
        ttk.Label(imu_frame, text="ΔYaw:").grid(row=11, column=2, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_yaw_delta).grid(row=11, column=3, sticky="w")

        pid_cols = ttk.Frame(tab_pid)
        pid_cols.pack(fill=tk.BOTH, expand=True)
        pid_cols.columnconfigure(0, weight=1)
        pid_cols.columnconfigure(1, weight=1)
        pid_x_frame = ttk.LabelFrame(pid_cols, text="X轴", padding=8)
        pid_y_frame = ttk.LabelFrame(pid_cols, text="Y轴", padding=8)
        pid_x_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        pid_y_frame.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        for c in range(2):
            pid_x_frame.columnconfigure(c, weight=1)
            pid_y_frame.columnconfigure(c, weight=1)
        row_x = 0
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kP", self.kp_x, 0.0, 0.01)
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kI", self.ki_x, 0.0, 0.01)
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kD", self.kd_x, 0.0, 0.01)
        row_y = 0
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kP", self.kp_y, 0.0, 0.01)
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kI", self.ki_y, 0.0, 0.01)
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kD", self.kd_y, 0.0, 0.01)
        common = ttk.LabelFrame(tab_pid, text="通用", padding=8)
        common.pack(fill=tk.X, pady=(10, 0))
        common.columnconfigure(0, weight=1)
        common.columnconfigure(1, weight=1)
        r2 = 0
        r2 = self._grid_slider(common, r2, 0, "死区(像素)", self.error_deadband, 0.0, 30.0)
        ttk.Label(common, text="误差小于此值时不响应，避免抖动", font=("", 8), foreground="gray").grid(row=r2-1, column=2, sticky="w", padx=(6, 0))
        r2 = self._grid_slider(common, r2, 0, "最大角速度(度/秒)", self.max_delta_deg_per_sec, 1.0, 500.0)
        ttk.Label(common, text="限制舵机转动速度，10-60度/秒较平滑", font=("", 8), foreground="gray").grid(row=r2-1, column=2, sticky="w", padx=(6, 0))

        # 卡尔曼滤波参数配置
        kalman_frame = ttk.LabelFrame(tab_pid, text="卡尔曼滤波 (Kalman Filter)", padding=8)
        kalman_frame.pack(fill=tk.X, pady=(10, 0))
        kalman_frame.columnconfigure(0, weight=1)
        kalman_frame.columnconfigure(1, weight=1)
        rk = 0
        rk = self._grid_slider(kalman_frame, rk, 0, "过程噪声(运动不可预测性)", self.kalman_process_noise, 0.001, 0.05)
        ttk.Label(kalman_frame, text="越小:平滑但延迟大; 越大:响应快但易抖动", font=("", 8), foreground="gray", wraplength=180).grid(row=rk-1, column=2, sticky="w", padx=(6, 0))
        rk = self._grid_slider(kalman_frame, rk, 0, "测量噪声(检测结果不稳定性)", self.kalman_measurement_noise, 0.01, 0.05)
        ttk.Label(kalman_frame, text="越大:平滑防抖强; 越小:极度信任视觉检测", font=("", 8), foreground="gray", wraplength=180).grid(row=rk-1, column=2, sticky="w", padx=(6, 0))

        # 舵机范围配置
        servo_range = ttk.LabelFrame(tab_pid, text="舵机角度范围", padding=8)
        servo_range.pack(fill=tk.X, pady=(10, 0))
        # 让两列平均分配宽度
        servo_range.columnconfigure(0, weight=1, uniform="col")
        servo_range.columnconfigure(1, weight=1, uniform="col")
        r3 = 0
        # 修改 _grid_slider 的调用方式，让它在半宽中正常显示
        r3 = self._grid_slider(servo_range, r3, 0, "水平最小", self.pan_min, -90.0, 0.0, colspan=1)
        # 上一行的调用返回的是 r3+1，为了让最大和最小在同一行，我们需要把行号退回
        self._grid_slider(servo_range, r3-1, 1, "水平最大", self.pan_max, 0.0, 90.0, colspan=1)
        r3 = self._grid_slider(servo_range, r3, 0, "俯仰最小", self.tilt_min, -90.0, 0.0, colspan=1)
        self._grid_slider(servo_range, r3-1, 1, "俯仰最大", self.tilt_max, 0.0, 90.0, colspan=1)
        self.range_canvas_pan = tk.Canvas(servo_range, height=24, highlightthickness=1, highlightbackground="#cfcfcf")
        self.range_canvas_pan.grid(row=r3, column=0, columnspan=2, sticky="ew", pady=(4, 2))
        r3 += 1
        self.range_canvas_tilt = tk.Canvas(servo_range, height=24, highlightthickness=1, highlightbackground="#cfcfcf")
        self.range_canvas_tilt.grid(row=r3, column=0, columnspan=2, sticky="ew", pady=(2, 2))
        r3 += 1
        self.range_hint_var = tk.StringVar(value="")
        ttk.Label(servo_range, textvariable=self.range_hint_var, foreground="gray").grid(
            row=r3, column=0, columnspan=2, sticky="w", pady=(2, 0)
        )
        
        # 硬件边界显示
        hw_range = ttk.LabelFrame(tab_pid, text="舵机物理边界 (硬件读取)", padding=8)
        hw_range.pack(fill=tk.X, pady=(10, 0))
        hw_range.columnconfigure(0, weight=1)
        hw_range.columnconfigure(1, weight=1)
        
        self.str_hw_pan_min = tk.StringVar(value="-90.0°")
        self.str_hw_pan_max = tk.StringVar(value="90.0°")
        self.str_hw_tilt_min = tk.StringVar(value="-90.0°")
        self.str_hw_tilt_max = tk.StringVar(value="90.0°")

        ttk.Label(hw_range, text="水平最小:").grid(row=0, column=0, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_pan_min).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(hw_range, text="水平最大:").grid(row=0, column=2, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_pan_max).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(hw_range, text="俯仰最小:").grid(row=1, column=0, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_tilt_min).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(hw_range, text="俯仰最大:").grid(row=1, column=2, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_tilt_max).grid(row=1, column=3, sticky="w", padx=5)
        
        # 绑定变量，以便动态更新字符串
        def update_hw_labels(*args):
            self.str_hw_pan_min.set(f"{self.hw_pan_min.get():.1f}°")
            self.str_hw_pan_max.set(f"{self.hw_pan_max.get():.1f}°")
            self.str_hw_tilt_min.set(f"{self.hw_tilt_min.get():.1f}°")
            self.str_hw_tilt_max.set(f"{self.hw_tilt_max.get():.1f}°")
        
        self.hw_pan_min.trace_add("write", update_hw_labels)
        self.hw_pan_max.trace_add("write", update_hw_labels)
        self.hw_tilt_min.trace_add("write", update_hw_labels)
        self.hw_tilt_max.trace_add("write", update_hw_labels)
        update_hw_labels()

        def map_x(v, width):
            left = 8
            right = max(left + 1, width - 8)
            return left + (float(v) + 90.0) / 180.0 * (right - left)

        def draw_one(canvas, gui_min, gui_max, hw_min, hw_max):
            w = max(120, canvas.winfo_width())
            h = max(20, canvas.winfo_height())
            y = h / 2.0
            canvas.delete("all")
            x0 = map_x(-90.0, w)
            x1 = map_x(90.0, w)
            canvas.create_line(x0, y, x1, y, fill="#bbbbbb", width=2)
            ph0 = map_x(hw_min, w)
            ph1 = map_x(hw_max, w)
            canvas.create_line(ph0, y, ph1, y, fill="#2f7ed8", width=6)
            gh0 = map_x(gui_min, w)
            gh1 = map_x(gui_max, w)
            canvas.create_line(gh0, y, gh1, y, fill="#25a75a", width=4)
            ef0 = map_x(max(hw_min, gui_min), w)
            ef1 = map_x(min(hw_max, gui_max), w)
            canvas.create_line(ef0, y, ef1, y, fill="#e04f5f", width=2)

        def update_range_visual(*_args):
            pan_gui_min = float(self.pan_min.get())
            pan_gui_max = float(self.pan_max.get())
            tilt_gui_min = float(self.tilt_min.get())
            tilt_gui_max = float(self.tilt_max.get())
            pan_hw_min = float(self.hw_pan_min.get())
            pan_hw_max = float(self.hw_pan_max.get())
            tilt_hw_min = float(self.hw_tilt_min.get())
            tilt_hw_max = float(self.hw_tilt_max.get())
            draw_one(self.range_canvas_pan, pan_gui_min, pan_gui_max, pan_hw_min, pan_hw_max)
            draw_one(self.range_canvas_tilt, tilt_gui_min, tilt_gui_max, tilt_hw_min, tilt_hw_max)
            self.range_hint_var.set(
                "蓝=物理范围  绿=GUI范围  红=最终生效范围(交集)"
            )

        for var in [self.pan_min, self.pan_max, self.tilt_min, self.tilt_max, self.hw_pan_min, self.hw_pan_max, self.hw_tilt_min, self.hw_tilt_max]:
            var.trace_add("write", update_range_visual)
        self.range_canvas_pan.bind("<Configure>", update_range_visual)
        self.range_canvas_tilt.bind("<Configure>", update_range_visual)
        update_range_visual()

        tab_vision.columnconfigure(0, weight=1)
        tab_vision.columnconfigure(1, weight=1)
        left_vis = ttk.LabelFrame(tab_vision, text="霍夫", padding=8)
        right_vis = ttk.LabelFrame(tab_vision, text="偏置/模糊", padding=8)
        left_vis.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right_vis.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        for c in range(2):
            left_vis.columnconfigure(c, weight=1)
            right_vis.columnconfigure(c, weight=1)
        rv = 0
        rv = self._grid_slider(left_vis, rv, 0, "最小间距", self.min_dist, 10, 300)
        rv = self._grid_slider(left_vis, rv, 0, "参数1", self.param1, 50, 500)
        rv = self._grid_slider(left_vis, rv, 0, "参数2", self.param2, 5, 200)
        rv = self._grid_slider(left_vis, rv, 0, "最小半径", self.min_radius, 1, 300)
        rv = self._grid_slider(left_vis, rv, 0, "最大半径", self.max_radius, 1, 300)
        rv2 = 0
        rv2 = self._grid_slider(right_vis, rv2, 0, "模糊核大小", self.ksize, 3, 19)
        
        # 视差校正偏置
        ttk.Separator(right_vis, orient=tk.HORIZONTAL).grid(row=rv2, column=0, columnspan=2, sticky="ew", pady=(5, 5))
        rv2 += 1
        rv2 = self._grid_slider(right_vis, rv2, 0, "X偏置", self.x_bias, -200, 200)
        rv2 = self._grid_slider(right_vis, rv2, 0, "Y偏置", self.y_bias, -200, 200)
        ttk.Label(right_vis, text="校正激光与相机的物理视差。\n(指示模式下发现光斑后自动失效)", font=("", 8), foreground="gray").grid(row=rv2, column=0, columnspan=2, sticky="w", pady=(0, 5))
        rv2 += 1
        
        # 激光指示对准配置
        ttk.Separator(right_vis, orient=tk.HORIZONTAL).grid(row=rv2, column=0, columnspan=2, sticky="ew", pady=(5, 5))
        rv2 += 1
        ttk.Checkbutton(right_vis, text="启用激光指示对准", variable=self.laser_align_mode).grid(row=rv2, column=0, columnspan=2, sticky="w", pady=(5, 0))
        rv2 += 1
        ttk.Label(right_vis, text="(在ROI内寻找最亮光斑)", font=("", 8), foreground="gray").grid(row=rv2, column=0, columnspan=2, sticky="w", pady=(0, 5))
        rv2 += 1
        rv2 = self._grid_slider(right_vis, rv2, 0, "光斑二值化阈值", self.laser_threshold, 100, 255)
        ttk.Label(right_vis, text="大于该亮度的像素将被视为光斑", font=("", 8), foreground="gray").grid(row=rv2, column=0, columnspan=2, sticky="w", pady=(0, 5))

        tab_camera.columnconfigure(0, weight=1)
        cam = ttk.Frame(tab_camera)
        cam.pack(fill=tk.BOTH, expand=True)
        cam.columnconfigure(0, weight=1)
        rc = 0
        rc = self._grid_slider(cam, rc, 0, "相机FPS", self.camera_fps, 10, 120)
        rc = self._grid_slider(cam, rc, 0, "图像旋转(°)", self.image_rotate_deg, -30.0, 30.0)
        
        # 添加红色警告提示Label (初始隐藏或为空)
        self.fps_warning_var = tk.StringVar(value="")
        self.fps_warning_label = ttk.Label(cam, textvariable=self.fps_warning_var, foreground="red", font=("", 9, "bold"))
        self.fps_warning_label.grid(row=rc, column=0, sticky="w", pady=(0, 4))
        rc += 1
        
        ttk.Checkbutton(cam, text="自动曝光", variable=self.ae_enable).grid(row=rc, column=0, sticky="w", pady=(2, 8))
        rc += 1
        rc = self._grid_slider(cam, rc, 0, "曝光(us)", self.exposure_value, 100, 100000)
        rc = self._grid_slider(cam, rc, 0, "增益", self.analogue_gain, 1.0, 22.0)

    def _on_status_text_changed(self, *_args):
        widget = self.status_log_widget
        if widget is None:
            return
        try:
            msg = self.status_text.get()
        except Exception:
            return
        if msg is None:
            return
        line = f"{time.strftime('%H:%M:%S')}  {msg}\n"
        widget.configure(state=tk.NORMAL)
        widget.insert(tk.END, line)
        widget.see(tk.END)
        widget.configure(state=tk.DISABLED)

    def _grid_entry(self, parent, row, col, text, var, width=10):
        ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=(2, 2))
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="ew", pady=(2, 2))

    def _grid_slider(self, parent, row, col, text, var, low, high, colspan=2):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, columnspan=colspan, sticky="ew", pady=(2, 6), padx=(0, 6 if col == 0 and colspan == 1 else 0))
        frame.columnconfigure(0, weight=1)
        header = ttk.Frame(frame)
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text=text).grid(row=0, column=0, sticky="w")
        if isinstance(var, tk.IntVar):
            value_text = tk.StringVar(value=str(var.get()))
        else:
            value_text = tk.StringVar(value=f"{var.get():.4f}")
        ttk.Label(header, textvariable=value_text).grid(row=0, column=1, sticky="e")
        scale = ttk.Scale(frame, from_=low, to=high, variable=var)
        scale.grid(row=1, column=0, sticky="ew")

        def _on_change(*_):
            if isinstance(var, tk.IntVar):
                value_text.set(str(int(var.get())))
            else:
                value_text.set(f"{var.get():.4f}")

        var.trace_add("write", _on_change)

        # 添加键盘方向键支持
        # 为了让小数步进更精细，整数步进保持1
        if isinstance(var, tk.IntVar):
            step = 1
            big_step = max(1, int((high - low) / 10.0))
        else:
            # 对于小数，每次按键移动 1/200 的量，按上下键移动 1/20
            step = (high - low) / 200.0
            big_step = (high - low) / 20.0

        def _on_key(event):
            current = var.get()
            if event.keysym == "Left":
                new_val = max(low, current - step)
            elif event.keysym == "Right":
                new_val = min(high, current + step)
            elif event.keysym == "Down":
                new_val = max(low, current - big_step)
            elif event.keysym == "Up":
                new_val = min(high, current + big_step)
            elif event.keysym == "Home":
                new_val = low
            elif event.keysym == "End":
                new_val = high
            else:
                return
            var.set(new_val)
            # 阻止默认事件，防止双重触发
            return "break"

        scale.bind("<Left>", _on_key)
        scale.bind("<Right>", _on_key)
        scale.bind("<Up>", _on_key)
        scale.bind("<Down>", _on_key)
        scale.bind("<Home>", _on_key)
        scale.bind("<End>", _on_key)
        
        # 必须允许scale获取焦点才能接收键盘事件
        scale.bind("<Button-1>", lambda e: scale.focus_set())
        scale.bind("<FocusIn>", lambda e: scale.focus_set())

        return row + 1

    def _start_runtime(self):
        if self.running:
            return
        try:
            self.stop_event.clear()
            self.detect_stop_event.clear()
            self.worker_error = None
            self._update_settings_from_vars()
            self._ensure_camera()
            self.running = True
            
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.detect_thread = threading.Thread(target=self._detect_loop, daemon=True)
            self.detect_thread.start()
            self.after_id = self.root.after(30, self._ui_loop)
            self.status_text.set("检测中（未跟踪）")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"初始化失败: {exc}")
            messagebox.showerror("初始化失败", str(exc))

    def start(self):
        if not self.running:
            self._start_runtime()
        self.tracking_active = True
        self.pid_x.reset()
        self.pid_y.reset()
        self.kalman = Kalman2D()
        self.stab_pan_residual_deg = 0.0
        self.stab_tilt_residual_deg = 0.0
        self.stab_pan_filtered_deg = 0.0
        self.stab_tilt_filtered_deg = 0.0
        self.status_text.set("跟踪已开始")

    def stop(self):
        self.tracking_active = False
        if self.running:
            self.status_text.set("检测中（未跟踪）")
        # 停止时不再回正，直接原地保持

    def reset_axes(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.kalman = Kalman2D()
        with self.detect_lock:
            self.latest_detection = None
            self.latest_detection_time = 0.0
        # 复位时强制回正
        self._center_servos()
        
    def _center_servos(self):
        """将舵机回正到0度"""
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        self.stab_pan_residual_deg = 0.0
        self.stab_tilt_residual_deg = 0.0
        self.stab_pan_filtered_deg = 0.0
        self.stab_tilt_filtered_deg = 0.0
        if self.servo is not None:
            self.servo.set_angles(
                [
                    (self.active_pan_id, 0.0),
                    (self.active_tilt_id, 0.0),
                ]
            )
            self.servo.move_angle(wait=False)

    def _worker_loop(self):
        try:
            last_time = time.time()
            while not self.stop_event.is_set():
                loop_start = time.time()
                dt = max(loop_start - last_time, 1e-4)
                last_time = loop_start

                if self.picam2 is None:
                    time.sleep(0.01)
                    try:
                        self._ensure_camera()
                    except Exception as exc:
                        self.worker_error = str(exc)
                        self.stop_event.set()
                        break
                    continue

                s = self._get_settings()
                if self.servo is not None:
                    self.servo.moving_time = max(0, int(s["move_time_ms"]))
                self._sync_camera_controls(s["ae_enable"], s["exposure"], s["gain"], s["camera_fps"])

                frame_rgb = self.picam2.capture_array()
                rotate_deg = float(s.get("image_rotate_deg", 0.0))
                if abs(rotate_deg) > 1e-3:
                    h0, w0 = frame_rgb.shape[:2]
                    c0 = (w0 * 0.5, h0 * 0.5)
                    rot_m = cv2.getRotationMatrix2D(c0, rotate_deg, 1.0)
                    frame_rgb = cv2.warpAffine(
                        frame_rgb,
                        rot_m,
                        (w0, h0),
                        flags=cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT,
                        borderValue=(0, 0, 0),
                    )
                h, w = frame_rgb.shape[:2]
                center_x, center_y = w // 2, h // 2
                with self.detect_lock:
                    self.latest_frame = (frame_rgb, s)
                    self.latest_frame_id += 1
                    detection = self.latest_detection
                    green_data = self.latest_green_channel
                    detection_age = time.time() - self.latest_detection_time if self.latest_detection_time > 0 else None
                if detection_age is None or detection_age > self.detect_stale_sec:
                    detection = None

                measurement = None
                radius = 0
                circle_found = detection is not None
                if detection is not None:
                    measurement = (float(detection[0]), float(detection[1]))
                    radius = int(detection[2])

                kalman_out = self.kalman.update(measurement)
                if kalman_out is None:
                    filtered_x, filtered_y = float(center_x), float(center_y)
                    pred_x, pred_y = float(center_x), float(center_y)
                else:
                    filtered_x, filtered_y, pred_x, pred_y = kalman_out
                    
                # 目标圆心坐标（这里暂不加 bias，后面根据对准模式决定）
                target_x = filtered_x
                target_y = filtered_y

                if not s["track_enabled"]:
                    target_x = float(center_x)
                    target_y = float(center_y)

                self.pid_x.set_gains(s["kp_x"], s["ki_x"], s["kd_x"])
                self.pid_y.set_gains(s["kp_y"], s["ki_y"], s["kd_y"])
                
                self.kalman.update_params(s["kalman_process_noise"], s["kalman_measurement_noise"])

                # 处理激光对准逻辑
                laser_spot_display = None
                laser_found = False
                laser_binary_display = None # 用于保存要显示的二值化图像
                
                # 判断当前是否满足指示对准的前提条件：
                # 1. 勾选了启用激光对准模式
                # 2. 必须先找到了圆形标志物 (circle_found) 才能谈“进入圆环内部”
                if s.get("laser_align_mode", False) and green_data is not None:
                    # green_data 现在是从 detect_loop 传过来的 5 个元素的元组
                    blurred_green, blurred_red, offset_x, offset_y, scale = green_data
                    _, binary = cv2.threshold(blurred_red, s.get("laser_threshold", 240), 255, cv2.THRESH_BINARY)
                    laser_binary_display = binary # 保存二值化结果
                    
                    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                    
                    if num_labels > 1:
                        # 找到了候选激光光斑
                        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
                        cx, cy = centroids[largest_label]
                        
                        # 转换回全图坐标
                        laser_x = (cx / scale) + offset_x
                        laser_y = (cy / scale) + offset_y
                        laser_spot_display = (laser_x, laser_y)
                        
                        if circle_found:
                            # 计算光斑到目标圆心的距离
                            dist_to_center = math.hypot(laser_x - target_x, laser_y - target_y)
                            
                            # 只有当光斑距离圆心小于圆的半径时，才认为“进入了圆形标志物内部”
                            if dist_to_center <= radius:
                                self.laser_locked_in_circle = True
                            else:
                                # 如果跑出了圆环，且距离过大（可以加个阈值防抖，这里简单处理一旦出圆就取消锁定）
                                # 为了防止在边缘频繁跳变，这里设定：如果超出半径 1.5 倍，则解除锁定，回退到盲对准。
                                if dist_to_center > radius * 1.5:
                                    self.laser_locked_in_circle = False
                        
                        # 如果当前处于锁定状态（光斑在圆内，或者刚才在圆内还没跑太远），则激活真正的指示对准
                        if self.laser_locked_in_circle:
                            laser_found = True
                            # 在指示模式下：直接使用光斑坐标对比真实圆心计算误差。抛弃 bias。
                            error_x = laser_x - target_x
                            error_y = laser_y - target_y

                # 如果没有开启指示对准，或者开启了但【没找到光斑】或【光斑还未进入圆环内部】
                if not laser_found:
                    # 此时必须使用盲对准逻辑（即使画面上有光斑，只要没进圆，依然由相机中心+bias来控制）
                    target_x += float(s["x_bias"])
                    target_y += float(s["y_bias"])
                    
                    error_x = float(center_x) - target_x
                    error_y = float(center_y) - target_y

                deadband = float(s["deadband"])
                if abs(error_x) < deadband:
                    error_x = 0.0
                if abs(error_y) < deadband:
                    error_y = 0.0
                if not circle_found:
                    error_x = 0.0
                    error_y = 0.0
                    self.pid_x.reset()
                    self.pid_y.reset()

                max_step = float(s["max_delta_deg_per_sec"]) * dt

                do_track = self.tracking_active
                do_stab = bool(s.get("auto_stabilize", False))
                if do_track or do_stab:
                    try:
                        # 每次循环都同步角度范围到舵机驱动层（确保GUI修改立即生效）
                        self._ensure_servo()
                    except Exception as exc:
                        self.worker_error = str(exc)
                        self.stop_event.set()
                        break

                stab_pan = 0.0
                stab_tilt = 0.0
                if do_stab:
                    try:
                        self._ensure_imu()
                        imu_state = self.imu.get_state() if self.imu is not None else None
                    except Exception as exc:
                        self.worker_error = f"IMU read failed: {exc}"
                        self.stop_event.set()
                        break
                    if imu_state is not None and imu_state.last_update > 0:
                        pitch_err = self._angle_diff_deg(imu_state.pitch_deg, self.imu_zero_pitch)
                        yaw_err = self._angle_diff_deg(imu_state.yaw_deg, self.imu_zero_yaw)
                        pitch_deadband = max(0.0, float(s.get("stab_pitch_deadband_deg", 0.6)))
                        if abs(pitch_err) < pitch_deadband:
                            pitch_err = 0.0
                        tilt_limit = max(0.0, float(s.get("stab_tilt_limit_deg", 8.0)))
                        stab_tilt_target = pitch_err * float(s.get("stab_gain_pitch", 1.0))
                        stab_tilt_target = max(-tilt_limit, min(tilt_limit, stab_tilt_target))
                        tilt_alpha = float(s.get("stab_tilt_alpha", 0.35))
                        tilt_alpha = max(0.0, min(1.0, tilt_alpha))
                        tilt_filtered_target = self.stab_tilt_filtered_deg + tilt_alpha * (stab_tilt_target - self.stab_tilt_filtered_deg)
                        tilt_rate_limit = max(0.0, float(s.get("stab_tilt_rate_limit_deg_per_s", 120.0)))
                        tilt_max_delta = tilt_rate_limit * dt
                        tilt_delta_filtered = tilt_filtered_target - self.stab_tilt_filtered_deg
                        if tilt_delta_filtered > tilt_max_delta:
                            tilt_delta_filtered = tilt_max_delta
                        elif tilt_delta_filtered < -tilt_max_delta:
                            tilt_delta_filtered = -tilt_max_delta
                        self.stab_tilt_filtered_deg = self.stab_tilt_filtered_deg + tilt_delta_filtered
                        stab_tilt_raw = self.stab_tilt_filtered_deg
                        yaw_deadband = max(0.0, float(s.get("stab_yaw_deadband_deg", 0.6)))
                        if abs(yaw_err) < yaw_deadband:
                            yaw_err = 0.0
                        pan_limit = max(0.0, float(s.get("stab_pan_limit_deg", 8.0)))
                        stab_pan_target = -yaw_err * float(s.get("stab_gain_yaw", 1.0))
                        stab_pan_target = max(-pan_limit, min(pan_limit, stab_pan_target))
                        alpha = float(s.get("stab_pan_alpha", 0.35))
                        alpha = max(0.0, min(1.0, alpha))
                        filtered_target = self.stab_pan_filtered_deg + alpha * (stab_pan_target - self.stab_pan_filtered_deg)
                        rate_limit = max(0.0, float(s.get("stab_pan_rate_limit_deg_per_s", 120.0)))
                        max_delta = rate_limit * dt
                        delta_filtered = filtered_target - self.stab_pan_filtered_deg
                        if delta_filtered > max_delta:
                            delta_filtered = max_delta
                        elif delta_filtered < -max_delta:
                            delta_filtered = -max_delta
                        self.stab_pan_filtered_deg = self.stab_pan_filtered_deg + delta_filtered
                        stab_pan_raw = self.stab_pan_filtered_deg
                        stab_tilt = self._quantize_to_servo_step_deg(stab_tilt_raw, axis="tilt")
                        stab_pan = self._quantize_to_servo_step_deg(stab_pan_raw, axis="pan")
                        self.latest_imu = (
                            float(imu_state.pitch_deg),
                            float(imu_state.yaw_deg),
                            float(time.time() - imu_state.last_update),
                            float(pitch_err),
                            float(yaw_err),
                        )
                # 使用GUI配置和硬件物理边界的交集作为最终限制
                # max(硬件最小, GUI最小) 和 min(硬件最大, GUI最大)
                pan_min = max(float(s.get("hw_pan_min", -90.0)), float(s.get("pan_min", -90.0)))
                pan_max = min(float(s.get("hw_pan_max", 90.0)), float(s.get("pan_max", 90.0)))
                tilt_min = max(float(s.get("hw_tilt_min", -90.0)), float(s.get("tilt_min", -90.0)))
                tilt_max = min(float(s.get("hw_tilt_max", 90.0)), float(s.get("tilt_max", 90.0)))
                gui_pan_min = float(s.get("pan_min", -90.0))
                gui_pan_max = float(s.get("pan_max", 90.0))
                gui_tilt_min = float(s.get("tilt_min", -90.0))
                gui_tilt_max = float(s.get("tilt_max", 90.0))
                hw_pan_min = float(s.get("hw_pan_min", -90.0))
                hw_pan_max = float(s.get("hw_pan_max", 90.0))
                hw_tilt_min = float(s.get("hw_tilt_min", -90.0))
                hw_tilt_max = float(s.get("hw_tilt_max", 90.0))

                # 更新前检测是否已经在边界，用于限制PID方向
                pan_at_min_before = self.current_pan_angle <= pan_min
                pan_at_max_before = self.current_pan_angle >= pan_max
                tilt_at_min_before = self.current_tilt_angle <= tilt_min
                tilt_at_max_before = self.current_tilt_angle >= tilt_max

                if do_track and s["pan_enabled"]:
                    delta_x = self.pid_x.update(error_x, dt=dt)
                    # 边界限制与抗积分饱和 (Anti-windup)
                    # 如果到达边界且PID试图继续向边界外移动，则禁止移动并清零积分项防止累积
                    if pan_at_min_before and delta_x < 0:
                        delta_x = 0
                        self.pid_x.i_term = 0.0
                    if pan_at_max_before and delta_x > 0:
                        delta_x = 0
                        self.pid_x.i_term = 0.0
                    desired_pan = self.current_pan_angle + delta_x
                    step_pan = max(-max_step, min(max_step, desired_pan - self.current_pan_angle))
                    self.current_pan_angle = max(pan_min, min(pan_max, self.current_pan_angle + step_pan))

                if do_track and s["tilt_enabled"]:
                    delta_y = self.pid_y.update(error_y, dt=dt)
                    # 边界限制与抗积分饱和 (Anti-windup)
                    if tilt_at_min_before and delta_y < 0:
                        delta_y = 0
                        self.pid_y.i_term = 0.0
                    if tilt_at_max_before and delta_y > 0:
                        delta_y = 0
                        self.pid_y.i_term = 0.0
                    desired_tilt = self.current_tilt_angle + delta_y
                    step_tilt = max(-max_step, min(max_step, desired_tilt - self.current_tilt_angle))
                    self.current_tilt_angle = max(tilt_min, min(tilt_max, self.current_tilt_angle + step_tilt))

                # 更新后的跟踪输出边界（不含IMU补偿）
                pan_at_min = self.current_pan_angle <= pan_min
                pan_at_max = self.current_pan_angle >= pan_max
                tilt_at_min = self.current_tilt_angle <= tilt_min
                tilt_at_max = self.current_tilt_angle >= tilt_max
                bound_reason_msgs = []

                if (do_track and (s["pan_enabled"] or s["tilt_enabled"])) or do_stab:
                    out_pan = self.current_pan_angle
                    out_tilt = self.current_tilt_angle
                    if s["pan_enabled"]:
                        pan_comp_min = pan_min - out_pan
                        pan_comp_max = pan_max - out_pan
                        stab_pan = max(pan_comp_min, min(pan_comp_max, stab_pan))
                        out_pan = out_pan + stab_pan
                    if s["tilt_enabled"]:
                        tilt_comp_min = tilt_min - out_tilt
                        tilt_comp_max = tilt_max - out_tilt
                        stab_tilt = max(tilt_comp_min, min(tilt_comp_max, stab_tilt))
                        out_tilt = out_tilt + stab_tilt
                    out_pan = max(pan_min, min(pan_max, out_pan))
                    out_tilt = max(tilt_min, min(tilt_max, out_tilt))

                    # 使用最终下发给舵机的角度计算边界提示，确保IMU稳定触发时也正确显示
                    pan_at_min = out_pan <= pan_min
                    pan_at_max = out_pan >= pan_max
                    tilt_at_min = out_tilt <= tilt_min
                    tilt_at_max = out_tilt >= tilt_max

                    if pan_at_min:
                        if gui_pan_min >= hw_pan_min:
                            bound_reason_msgs.append("水平下限受GUI限制")
                        else:
                            bound_reason_msgs.append("水平下限受物理限制")
                    if pan_at_max:
                        if gui_pan_max <= hw_pan_max:
                            bound_reason_msgs.append("水平上限受GUI限制")
                        else:
                            bound_reason_msgs.append("水平上限受物理限制")
                    if tilt_at_min:
                        if gui_tilt_min >= hw_tilt_min:
                            bound_reason_msgs.append("俯仰下限受GUI限制")
                        else:
                            bound_reason_msgs.append("俯仰下限受物理限制")
                    if tilt_at_max:
                        if gui_tilt_max <= hw_tilt_max:
                            bound_reason_msgs.append("俯仰上限受GUI限制")
                        else:
                            bound_reason_msgs.append("俯仰上限受物理限制")
                    self.servo.set_angles(
                        [
                            (self.active_pan_id, out_pan),
                            (self.active_tilt_id, out_tilt),
                        ]
                    )
                    self.servo.move_angle(wait=False)

                # 获取当前激光测距数据 (如果在死区内，且距离上次查询超过1秒)
                current_distance = -1.0
                if self.laser_ranger is not None and abs(error_x) <= deadband and abs(error_y) <= deadband and circle_found:
                    now = time.time()
                    if now - self.last_laser_query_time >= 1.0:
                        self.last_laser_query_time = now
                        # 单次发起查询
                        if self.laser_ranger.query_once():
                            dist_m = self.laser_ranger.distance_m
                            print(f"[测距结果] 目标对准！当前距离: {dist_m:.3f} m, 信号强度: {self.laser_ranger.signal_strength}")
                            
                    # 获取最新距离数据用于显示
                    if len(self.laser_ranger.distance_m_data) > 0:
                        current_distance = self.laser_ranger.distance_m_data[-1]

                servo_status = self.latest_servo_status
                now = time.time()
                if self.servo is not None and now - self.last_servo_status_time >= 0.5:
                    try:
                        if s.get("servo_mode") == "控制板":
                            pos_map = self.servo.read_positions()
                            pan_pos = pos_map.get(self.active_pan_id)
                            tilt_pos = pos_map.get(self.active_tilt_id)
                            voltage = self.servo.read_voltage_mv()
                        else:
                            pos_list = self.servo.read_servos_angle()
                            if self.active_pan_id == self.active_tilt_id and len(pos_list) >= 2:
                                pan_pos = pos_list[0][1]
                                tilt_pos = pos_list[1][1]
                            else:
                                pos_map = {sid: pos for sid, pos in pos_list}
                                pan_pos = pos_map.get(self.active_pan_id)
                                tilt_pos = pos_map.get(self.active_tilt_id)
                            voltage = None
                        servo_status = (s.get("servo_mode"), pan_pos, tilt_pos, voltage)
                        self.latest_servo_status = servo_status
                        self.last_servo_status_time = now
                    except Exception:
                        servo_status = self.latest_servo_status

                # 创建双屏显示的原始数据，转移到 UI 线程去拼接
                # 这里只发送原始数据给 UI 线程，彻底解放控制线程的耗时
                payload = (
                    frame_rgb, # 原始 BGR 图像
                    dt,
                    (error_x, error_y),
                    (self.current_pan_angle, self.current_tilt_angle),
                    circle_found,
                    radius,
                    do_track,
                    self.laser_locked_in_circle,
                    green_data, # 包含 (blurred_green, blurred_red, offset_x, offset_y, scale)
                    detection,
                    (target_x, target_y),
                    (pred_x, pred_y),
                    laser_spot_display,
                    (pan_at_min, pan_at_max, tilt_at_min, tilt_at_max),
                    " | ".join(bound_reason_msgs),
                    s.get("laser_threshold", 240),
                    deadband,
                    current_distance,
                    servo_status
                )
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait(payload)
                except Exception:
                    pass

                period_ms = max(1, int(s["control_period_ms"]))
                elapsed_ms = int((time.time() - loop_start) * 1000)
                sleep_ms = max(0, period_ms - elapsed_ms)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
        except Exception as exc:
            import traceback
            print(f"[ERROR] worker loop exception: {exc}")
            traceback.print_exc()
            self.worker_error = str(exc)
            self.stop_event.set()

    def _detect_loop(self):
        last_processed_id = 0
        try:
            while not self.detect_stop_event.is_set():
                with self.detect_lock:
                    frame_bundle = self.latest_frame
                    frame_id = self.latest_frame_id
                if frame_bundle is None or frame_id == last_processed_id:
                    time.sleep(0.002)
                    continue
                frame_rgb, s = frame_bundle
                roi = None
                with self.detect_lock:
                    last_det = self.latest_detection
                    last_det_time = self.latest_detection_time
                if last_det is not None and (time.time() - last_det_time) <= self.detect_stale_sec:
                    h, w = frame_rgb.shape[:2]
                    margin = max(80, int(last_det[2] * 2.5))
                    x0 = max(0, int(last_det[0] - margin))
                    y0 = max(0, int(last_det[1] - margin))
                    x1 = min(w, int(last_det[0] + margin))
                    y1 = min(h, int(last_det[1] + margin))
                    if x1 - x0 >= 20 and y1 - y0 >= 20:
                        roi = (x0, y0, x1, y1)
                detection, blurred_green, blurred_red, offset_x, offset_y, scale = self._detect_circle(
                    frame_rgb,
                    ksize=s["ksize"],
                    min_dist=s["min_dist"],
                    param1=s["param1"],
                    param2=s["param2"],
                    min_radius=s["min_radius"],
                    max_radius=s["max_radius"],
                    roi=roi,
                )
                
                # 应用EMA低通滤波稳定检测结果
                if detection is not None:
                    if self.smoothed_detection is None:
                        self.smoothed_detection = list(detection)
                    else:
                        alpha = self.ema_alpha
                        self.smoothed_detection[0] = alpha * detection[0] + (1 - alpha) * self.smoothed_detection[0]
                        self.smoothed_detection[1] = alpha * detection[1] + (1 - alpha) * self.smoothed_detection[1]
                        self.smoothed_detection[2] = alpha * detection[2] + (1 - alpha) * self.smoothed_detection[2]
                    detection_to_save = tuple(self.smoothed_detection)
                else:
                    self.smoothed_detection = None
                    detection_to_save = None

                with self.detect_lock:
                    self.latest_detection = detection_to_save
                    self.latest_detection_time = time.time()
                    self.latest_green_channel = (blurred_green, blurred_red, offset_x, offset_y, scale)
                    last_processed_id = frame_id
        except Exception as exc:
            import traceback
            print(f"[ERROR] detect loop exception: {exc}")
            traceback.print_exc()
            self.worker_error = str(exc)
            self.stop_event.set()

    def _ui_loop(self):
        if not self.running:
            self.after_id = self.root.after(100, self._ui_loop)
            return
        self._update_settings_from_vars()
        if self.worker_error is not None:
            self.status_text.set(f"工作线程错误: {self.worker_error}")
            self.after_id = self.root.after(200, self._ui_loop)
            return
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except Exception:
            pass

        if latest is not None:
            (
                frame_rgb,
                dt,
                (error_x, error_y),
                (pan, tilt),
                circle_found,
                radius,
                do_track,
                laser_locked,
                green_data,
                detection,
                (target_x, target_y),
                (pred_x, pred_y),
                laser_spot_display,
                bounds,
                bound_reason,
                laser_threshold,
                deadband,
                current_distance,
                servo_status
            ) = latest
            
            h, w = frame_rgb.shape[:2]
            center_x, center_y = w // 2, h // 2
            
            # 在 UI 线程中进行耗时的绘制和拼接
            self._draw_overlay(
                frame=frame_rgb,
                center=(center_x, center_y),
                detection=detection,
                target=(int(round(target_x)), int(round(target_y))),
                pred=(int(round(pred_x)), int(round(pred_y))),
                radius=radius,
                error=(error_x, error_y),
                dt=dt,
                bounds=bounds,
                laser_spot=laser_spot_display,
            )
            
            frame_rgb_disp = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
            if abs(error_x) <= deadband and abs(error_y) <= deadband and circle_found:
                cv2.putText(frame_rgb_disp, "ALIGNMENT COMPLETE", (10, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                if current_distance > 0:
                    cv2.putText(frame_rgb_disp, f"{current_distance:.3f} m", (10, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            
            if self.show_debug_panels.get() and green_data is not None:
                blurred_green, blurred_red, offset_x, offset_y, scale = green_data
                
                gh, gw = blurred_green.shape[:2]
                orig_w = int(gw / scale)
                orig_h = int(gh / scale)
                
                end_y = min(h, offset_y + orig_h)
                end_x = min(w, offset_x + orig_w)
                roi_h = end_y - offset_y
                roi_w = end_x - offset_x
                
                full_green = np.zeros_like(frame_rgb_disp)
                full_red = np.zeros_like(frame_rgb_disp)
                
                if roi_h > 0 and roi_w > 0:
                    green_resized_single = cv2.resize(blurred_green, (orig_w, orig_h))
                    red_resized_single = cv2.resize(blurred_red, (orig_w, orig_h))
                    
                    full_green[offset_y:end_y, offset_x:end_x, 1] = green_resized_single[:roi_h, :roi_w]
                    full_red[offset_y:end_y, offset_x:end_x, 0] = red_resized_single[:roi_h, :roi_w]
                
                if detection is not None:
                    x, y, r = detection
                    x, y, r = int(round(x)), int(round(y)), int(round(r))
                    cv2.circle(full_green, (x, y), 3, (255, 0, 0), -1)
                    cv2.circle(full_green, (x, y), r, (255, 0, 0), 2)
                    
                if laser_spot_display is not None:
                    lx, ly = int(round(laser_spot_display[0])), int(round(laser_spot_display[1]))
                    cv2.line(full_green, (lx-10, ly), (lx+10, ly), (255, 255, 0), 2)
                    cv2.line(full_green, (lx, ly-10), (lx, ly+10), (255, 255, 0), 2)
                    cv2.putText(full_green, "Laser", (lx+10, ly-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                cv2.putText(full_green, "Green Channel (Processed)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(full_red, "Red Channel (Processed)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                
                if self.laser_align_mode.get():
                    _, binary = cv2.threshold(blurred_red, laser_threshold, 255, cv2.THRESH_BINARY)
                    full_bin = np.zeros_like(frame_rgb_disp)
                    bin_resized = cv2.resize(binary, (orig_w, orig_h))
                    if roi_h > 0 and roi_w > 0:
                        full_bin[offset_y:end_y, offset_x:end_x, 0] = bin_resized[:roi_h, :roi_w]
                    
                    if laser_locked:
                        cv2.putText(full_bin, "Laser Binary Mask [LOCKED]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        # 同时在原图的右下角也写上锁定状态
                        cv2.putText(frame_rgb_disp, "Laser: LOCKED", (w - 250, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    else:
                        cv2.putText(full_bin, "Laser Binary Mask [SEARCHING]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                        cv2.putText(frame_rgb_disp, "Laser: SEARCHING", (w - 300, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
                else:
                    full_bin = np.zeros_like(frame_rgb_disp)
                    cv2.putText(full_bin, "Laser Binary (Disabled)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                    cv2.putText(frame_rgb_disp, "Laser: BLIND ALIGN", (w - 320, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                
                top_row = np.hstack((frame_rgb_disp, full_green))
                bottom_row = np.hstack((full_red, full_bin))
                frame_rgb_show = np.vstack((top_row, bottom_row))
            else:
                frame_rgb_show = frame_rgb_disp

            image = Image.fromarray(frame_rgb_show)
            photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            
            # 计算真实相机FPS
            current_time = time.time()
            with self.detect_lock:
                current_frame_id = self.latest_frame_id
            if current_frame_id > self.last_cam_frame_id:
                if self.last_cam_time > 0:
                    cam_dt = current_time - self.last_cam_time
                    self.fps_cam = (current_frame_id - self.last_cam_frame_id) / cam_dt
                self.last_cam_time = current_time
                self.last_cam_frame_id = current_frame_id

            if dt > 0:
                self.fps_ctrl = 1.0 / dt
                
            status_msg = f"相机FPS={self.fps_cam:.1f}  控制Hz={self.fps_ctrl:.1f}  水平={pan:.2f}  俯仰={tilt:.2f}  误差X={error_x:.1f}  误差Y={error_y:.1f}  圆={int(circle_found)}  半径={radius}  跟踪={int(do_track)}"
            if bound_reason:
                status_msg += f"  超限:{bound_reason}"
            imu_snapshot = self.latest_imu
            if imu_snapshot is not None:
                pitch_deg, yaw_deg, age_sec, pitch_delta, yaw_delta = imu_snapshot
                self.imu_status_pitch.set(f"{pitch_deg:+.2f}")
                self.imu_status_yaw.set(f"{yaw_deg:+.2f}")
                self.imu_status_age.set(f"{age_sec:.2f}s")
                self.imu_status_pitch_base.set(f"{self.imu_zero_pitch:+.2f}")
                self.imu_status_yaw_base.set(f"{self.imu_zero_yaw:+.2f}")
                self.imu_status_pitch_delta.set(f"{pitch_delta:+.2f}")
                self.imu_status_yaw_delta.set(f"{yaw_delta:+.2f}")
                status_msg += f"  稳定={int(self.auto_stabilize.get())}  IMU(p={pitch_deg:+.1f},y={yaw_deg:+.1f},dp={pitch_delta:+.1f},dy={yaw_delta:+.1f})"
            else:
                self.imu_status_pitch.set("-")
                self.imu_status_yaw.set("-")
                self.imu_status_age.set("-")
                self.imu_status_pitch_base.set(f"{self.imu_zero_pitch:+.2f}")
                self.imu_status_yaw_base.set(f"{self.imu_zero_yaw:+.2f}")
                self.imu_status_pitch_delta.set("-")
                self.imu_status_yaw_delta.set("-")
            if self.laser_align_mode.get():
                if laser_locked:
                    status_msg += "  [指示对准: 已锁定光斑]"
                else:
                    status_msg += "  [盲对准: 寻找光斑中...]"
                    
            self.status_text.set(status_msg)
            self._update_servo_status_labels(servo_status)
            
            # 检测并更新FPS与曝光冲突警告
            target_fps = self.camera_fps.get()
            current_exposure = self.exposure_value.get()
            is_auto_exposure = self.ae_enable.get()
            
            if target_fps > 0 and not is_auto_exposure:
                frame_duration_us = int(1000000 / target_fps)
                if current_exposure > frame_duration_us:
                    actual_max_fps = int(1000000 / current_exposure)
                    self.fps_warning_var.set(f"⚠️ 曝光过长! 帧率被强制降至 ≤ {actual_max_fps} FPS")
                else:
                    self.fps_warning_var.set("")
            else:
                self.fps_warning_var.set("")

        self.after_id = self.root.after(30, self._ui_loop)

    def _sync_camera_controls(self, ae_enable, exposure, gain, target_fps):
        # 检查是否有变化
        ae_changed = self.last_ae_enable != ae_enable
        exposure_changed = self.last_exposure != exposure
        gain_changed = self.last_gain != gain
        fps_changed = getattr(self, 'last_fps', None) != target_fps
        
        if not ae_changed and not exposure_changed and not gain_changed and not fps_changed:
            return
            
        controls_to_set = {}
        
        # 处理帧率修改，并保证“曝光优先于FPS”
        # 如果是手动曝光模式，且设置的曝光时间大于目标FPS的单帧时间，则必须放宽FrameDurationLimits
        if target_fps > 0:
            frame_duration_us = int(1000000 / target_fps)
            if not ae_enable and exposure > frame_duration_us:
                # 曝光时间超出了当前FPS的物理极限，延长帧间距以满足曝光
                actual_duration = int(exposure)
            else:
                actual_duration = frame_duration_us
                
            if fps_changed or (not ae_enable and exposure_changed):
                controls_to_set["FrameDurationLimits"] = (actual_duration, actual_duration)
                self.last_fps = target_fps
        
        # 自动曝光关闭时，同时设置AeEnable和曝光参数
        if not ae_enable:
            if ae_changed:
                controls_to_set["AeEnable"] = False
            if exposure_changed or ae_changed:
                controls_to_set["ExposureTime"] = int(exposure)
            if gain_changed or ae_changed:
                controls_to_set["AnalogueGain"] = float(gain)
        else:
            # 自动曝光开启时，只设置AeEnable
            if ae_changed:
                controls_to_set["AeEnable"] = True
                
        if controls_to_set:
            self.picam2.set_controls(controls_to_set)
        
        self.last_ae_enable = ae_enable
        self.last_exposure = exposure
        self.last_gain = gain

    def _ensure_camera(self):
        if self.picam2 is not None:
            return
        settings = self._get_settings()
        self.picam2 = Picamera2()
        self.picam2.start_preview(Preview.NULL)
        framerate = settings.get("camera_fps", 60)
        frame_duration = int(1000000 / framerate)
        config = self.picam2.create_video_configuration(
            controls={"FrameDurationLimits": (frame_duration, frame_duration)}
        )
        config["main"]["format"] = "RGB888"
        config["main"]["size"] = (640, 640)
        self.picam2.align_configuration(config)
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)
        self.last_ae_enable = None
        self.last_exposure = None
        self.last_gain = None
        self.last_fps = framerate

    @staticmethod
    def _angle_diff_deg(a, b):
        d = float(a) - float(b)
        while d > 180.0:
            d -= 360.0
        while d < -180.0:
            d += 360.0
        return d

    def _quantize_to_servo_step_deg(self, angle_deg: float, axis: str = "pan") -> float:
        raw = float(angle_deg)
        if axis == "pan":
            total = raw + float(self.stab_pan_residual_deg)
        else:
            total = raw + float(self.stab_tilt_residual_deg)
        step = float(self.servo_deg_per_step)
        steps = int(round(total / step))
        quantized = steps * step
        residual = total - quantized
        if axis == "pan":
            self.stab_pan_residual_deg = residual
        else:
            self.stab_tilt_residual_deg = residual
        return quantized

    def _ensure_imu(self):
        if self.imu is not None:
            return
        settings = self._get_settings()
        self.imu = IMUReader(
            port=settings.get("imu_port", self.imu_port.get()),
            baudrate=int(settings.get("imu_baudrate", self.imu_baudrate.get())),
            timeout=0.1,
            debug=False,
        )
        try:
            self.imu.configure_output(output_mask=0x001E, rate_code=0x08)
            self.imu.set_algorithm_mode(bool(settings.get("imu_use_6axis", self.imu_use_6axis.get())))
            hz = int(settings.get("imu_output_hz", self.imu_output_hz.get()))
            self.imu.set_output_rate_hz(hz)
        except Exception:
            pass
        self.imu.start()

    def _close_imu(self):
        if self.imu is None:
            return
        try:
            self.imu.close()
        except Exception:
            pass
        self.imu = None

    def _zero_imu(self):
        try:
            self._ensure_imu()
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU错误: {exc}")
            return
        deadline = time.time() + 1.5
        pitches = []
        yaw_sin = 0.0
        yaw_cos = 0.0
        while time.time() < deadline and len(pitches) < 40:
            try:
                state = self.imu.get_state()
                if state.last_update <= 0:
                    time.sleep(0.02)
                    continue
                pitches.append(float(state.pitch_deg))
                yaw_rad = math.radians(float(state.yaw_deg))
                yaw_sin += math.sin(yaw_rad)
                yaw_cos += math.cos(yaw_rad)
            except Exception:
                pass
            time.sleep(0.02)
        if len(pitches) < 5:
            self.status_text.set("IMU置零失败: 无数据")
            return
        self.imu_zero_pitch = float(sum(pitches) / len(pitches))
        self.imu_zero_yaw = float(math.degrees(math.atan2(yaw_sin, yaw_cos)))
        self.stab_pan_residual_deg = 0.0
        self.stab_tilt_residual_deg = 0.0
        self.stab_pan_filtered_deg = 0.0
        self.stab_tilt_filtered_deg = 0.0
        self.imu_status_pitch_base.set(f"{self.imu_zero_pitch:+.2f}")
        self.imu_status_yaw_base.set(f"{self.imu_zero_yaw:+.2f}")
        self.imu_status_pitch_delta.set("+0.00")
        self.imu_status_yaw_delta.set("+0.00")
        self.status_text.set("IMU已置零(平均)")

    def _apply_imu_output_rate(self):
        try:
            self._ensure_imu()
            hz = int(self.imu_output_hz.get())
            self.imu.set_output_rate_hz(hz)
            self.status_text.set(f"IMU输出速率已设置为 {hz}Hz")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU配置失败: {exc}")

    def _apply_imu_algorithm_mode(self):
        try:
            self._ensure_imu()
            use_6axis = bool(self.imu_use_6axis.get())
            self.imu.set_algorithm_mode(use_6axis)
            self.status_text.set("IMU算法已设置为6轴" if use_6axis else "IMU算法已设置为9轴")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU配置失败: {exc}")

    def _apply_imu_baudrate(self):
        try:
            self._ensure_imu()
            baud = int(self.imu_baudrate.get())
            self.imu.apply_baudrate(baud)
            self.status_text.set(f"IMU波特率已设置为 {baud}")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU配置失败: {exc}")

    def _apply_imu_offsets(self):
        try:
            self._ensure_imu()
            az_offset = float(self.imu_az_offset_g.get())
            if abs(az_offset) > 0.30:
                ok = messagebox.askyesno(
                    "确认AZ零偏",
                    f"当前 AZ(g) 零偏为 {az_offset:+.4f}g，绝对值较大。\n请确认你填写的是“相对理想值的误差”而不是直接去掉1g。\n仍然写入吗？",
                )
                if not ok:
                    self.status_text.set("已取消写入IMU零偏")
                    return
            self.imu.set_sensor_offsets(
                ax_g=float(self.imu_ax_offset_g.get()),
                ay_g=float(self.imu_ay_offset_g.get()),
                az_g=float(self.imu_az_offset_g.get()),
                gx_dps=float(self.imu_gx_offset_dps.get()),
                gy_dps=float(self.imu_gy_offset_dps.get()),
                gz_dps=float(self.imu_gz_offset_dps.get()),
                hx=int(self.imu_hx_offset.get()),
                hy=int(self.imu_hy_offset.get()),
                hz=int(self.imu_hz_offset.get()),
            )
            self.status_text.set("IMU零偏已写入并保存")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU零偏写入失败: {exc}")

    def _sample_fill_imu_offsets_flat(self):
        try:
            self._ensure_imu()
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU错误: {exc}")
            return
        deadline = time.time() + 1.2
        acc_x_vals = []
        acc_y_vals = []
        acc_z_vals = []
        gyro_x_vals = []
        gyro_y_vals = []
        gyro_z_vals = []
        mag_x_vals = []
        mag_y_vals = []
        mag_z_vals = []
        while time.time() < deadline and len(acc_x_vals) < 50:
            try:
                s = self.imu.get_state()
                if s.last_update <= 0:
                    time.sleep(0.01)
                    continue
                acc_x_vals.append(float(s.acc_x_g))
                acc_y_vals.append(float(s.acc_y_g))
                acc_z_vals.append(float(s.acc_z_g))
                gyro_x_vals.append(float(s.gyro_x_dps))
                gyro_y_vals.append(float(s.gyro_y_dps))
                gyro_z_vals.append(float(s.gyro_z_dps))
                mag_x_vals.append(float(getattr(s, "mag_x_raw", 0.0)))
                mag_y_vals.append(float(getattr(s, "mag_y_raw", 0.0)))
                mag_z_vals.append(float(getattr(s, "mag_z_raw", 0.0)))
            except Exception:
                pass
            time.sleep(0.02)
        if len(acc_x_vals) < 10:
            self.status_text.set("静置采样失败: IMU数据不足")
            return
        mean_ax = sum(acc_x_vals) / len(acc_x_vals)
        mean_ay = sum(acc_y_vals) / len(acc_y_vals)
        mean_az = sum(acc_z_vals) / len(acc_z_vals)
        mean_gx = sum(gyro_x_vals) / len(gyro_x_vals)
        mean_gy = sum(gyro_y_vals) / len(gyro_y_vals)
        mean_gz = sum(gyro_z_vals) / len(gyro_z_vals)
        mean_mx = sum(mag_x_vals) / len(mag_x_vals) if mag_x_vals else 0.0
        mean_my = sum(mag_y_vals) / len(mag_y_vals) if mag_y_vals else 0.0
        mean_mz = sum(mag_z_vals) / len(mag_z_vals) if mag_z_vals else 0.0
        az_ref = float(self.imu_az_reference_g.get())
        self.imu_ax_offset_g.set(-mean_ax)
        self.imu_ay_offset_g.set(-mean_ay)
        self.imu_az_offset_g.set(az_ref - mean_az)
        self.imu_gx_offset_dps.set(-mean_gx)
        self.imu_gy_offset_dps.set(-mean_gy)
        self.imu_gz_offset_dps.set(-mean_gz)
        self.imu_hx_offset.set(int(round(-mean_mx)))
        self.imu_hy_offset.set(int(round(-mean_my)))
        self.imu_hz_offset.set(int(round(-mean_mz)))
        self.status_text.set(
            f"已填入零偏: AX={-mean_ax:+.4f} AY={-mean_ay:+.4f} AZ={az_ref - mean_az:+.4f} "
            f"HX={int(round(-mean_mx))} HY={int(round(-mean_my))} HZ={int(round(-mean_mz))}"
        )

    def _open_imu_offsets_dialog(self):
        win = tk.Toplevel(self.root)
        win.title("IMU 零偏设置")
        win.transient(self.root)
        win.grab_set()
        frame = ttk.Frame(win, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        labels = [
            ("AX(g)", self.imu_ax_offset_g),
            ("AY(g)", self.imu_ay_offset_g),
            ("AZ(g)", self.imu_az_offset_g),
            ("GX(dps)", self.imu_gx_offset_dps),
            ("GY(dps)", self.imu_gy_offset_dps),
            ("GZ(dps)", self.imu_gz_offset_dps),
            ("HX", self.imu_hx_offset),
            ("HY", self.imu_hy_offset),
            ("HZ", self.imu_hz_offset),
        ]
        for i, (name, var) in enumerate(labels):
            ttk.Label(frame, text=name).grid(row=i, column=0, sticky="e", padx=(0, 8), pady=3)
            ttk.Entry(frame, textvariable=var, width=14).grid(row=i, column=1, sticky="w", pady=3)
        row = len(labels)
        ttk.Label(frame, text="AZ参考(g)").grid(row=row, column=0, sticky="e", padx=(0, 8), pady=(8, 3))
        ttk.Entry(frame, textvariable=self.imu_az_reference_g, width=14).grid(row=row, column=1, sticky="w", pady=(8, 3))
        row += 1
        ttk.Button(frame, text="静置采样填入零偏", command=self._sample_fill_imu_offsets_flat).grid(
            row=row, column=0, columnspan=2, sticky="ew", pady=(6, 0)
        )
        row += 1
        ttk.Label(
            frame,
            text="静置采样会同时填入HX/HY/HZ当前均值反向值；磁场零偏更建议多方向旋转后再精调。",
        ).grid(row=row, column=0, columnspan=2, sticky="w", pady=(6, 0))
        btns = ttk.Frame(frame)
        btns.grid(row=row + 1, column=0, columnspan=2, sticky="ew", pady=(10, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        ttk.Button(btns, text="写入零偏", command=self._apply_imu_offsets).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="关闭", command=win.destroy).grid(row=0, column=1, sticky="ew", padx=(8, 0))

    def _load_board_modules(self):
        if self._board_transport_cls is not None and self._board_driver_cls is not None:
            return
        base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bus_servo_ctrl_board")
        original_protocol = sys.modules.get("protocol")
        original_transport = sys.modules.get("transport")
        original_driver = sys.modules.get("driver")
        try:
            board_protocol = _load_module("_board_protocol", os.path.join(base_dir, "protocol.py"))
            sys.modules["protocol"] = board_protocol
            
            board_transport = _load_module("_board_transport", os.path.join(base_dir, "transport.py"))
            sys.modules["transport"] = board_transport
            
            board_driver = _load_module("_board_driver", os.path.join(base_dir, "driver.py"))
            sys.modules["driver"] = board_driver
        finally:
            if original_protocol is not None:
                sys.modules["protocol"] = original_protocol
            else:
                sys.modules.pop("protocol", None)
            if original_transport is not None:
                sys.modules["transport"] = original_transport
            else:
                sys.modules.pop("transport", None)
            if original_driver is not None:
                sys.modules["driver"] = original_driver
            else:
                sys.modules.pop("driver", None)
        self._board_transport_cls = board_transport.SerialTransport
        self._board_driver_cls = board_driver.BusServoBoardDriver

    def _get_bus_servo_cls(self):
        if self._bus_servo_cls is not None:
            return self._bus_servo_cls
        base_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 缓存环境
        original_protocol = sys.modules.get("protocol")
        original_transport = sys.modules.get("transport")
        original_driver = sys.modules.get("driver")
        
        try:
            bus_servo_module = _load_module("_bus_servo_module", os.path.join(base_dir, "bus_servo.py"))
            self._bus_servo_cls = bus_servo_module.BusServo
        finally:
            if original_protocol is not None:
                sys.modules["protocol"] = original_protocol
            else:
                sys.modules.pop("protocol", None)
            if original_transport is not None:
                sys.modules["transport"] = original_transport
            else:
                sys.modules.pop("transport", None)
            if original_driver is not None:
                sys.modules["driver"] = original_driver
            else:
                sys.modules.pop("driver", None)
                
        return self._bus_servo_cls

    def _get_brushless_motor_classes(self):
        if self._brushless_motor_config_cls is not None and self._brushless_motor_cls is not None:
            return self._brushless_motor_config_cls, self._brushless_motor_cls
        base_dir = os.path.dirname(os.path.abspath(__file__))
        module = _load_module(
            "_brushless_dual_driver_v1",
            os.path.join(base_dir, "brushless_motor", "dual_rs485_motor_driver_v1.py"),
        )
        self._brushless_motor_config_cls = module.MotorConfig
        self._brushless_motor_cls = module.LkMotor
        return self._brushless_motor_config_cls, self._brushless_motor_cls

    def _release_servo(self):
        if self.servo is not None:
            try:
                self.servo.cleanup()
            except Exception:
                try:
                    self.servo.close()
                except Exception:
                    pass
        self.servo = None
        self.latest_servo_status = None
        self.last_servo_status_time = 0.0
        self.servo_status_mode.set(self.servo_mode.get())
        self.servo_status_pan.set("-")
        self.servo_status_tilt.set("-")
        self.servo_status_voltage.set("-")

    def _on_servo_mode_change(self, *_):
        if self.servo_mode.get() == "控制板":
            self.baudrate.set(9600)
        elif self.servo_mode.get() == "无刷RS485":
            self.baudrate.set(1000000)
        else:
            self.baudrate.set(115200)
        self._refresh_servo_mode_ui()
        self._release_servo()

    def _refresh_servo_mode_ui(self):
        mode = self.servo_mode.get()
        is_brushless = mode == "无刷RS485"
        if hasattr(self, "port_label"):
            self.port_label.configure(text="旧串口(非无刷)" if is_brushless else "串口")
        if hasattr(self, "port_entry"):
            self.port_entry.configure(state=(tk.DISABLED if is_brushless else tk.NORMAL))
        if hasattr(self, "baudrate_label"):
            self.baudrate_label.configure(text="旧波特率(非无刷)" if is_brushless else "波特率")
        if hasattr(self, "baudrate_entry"):
            self.baudrate_entry.configure(state=(tk.DISABLED if is_brushless else tk.NORMAL))

    def _update_servo_status_labels(self, servo_status):
        if servo_status is None:
            self.servo_status_mode.set(self.servo_mode.get())
            self.servo_status_pan.set("-")
            self.servo_status_tilt.set("-")
            self.servo_status_voltage.set("-")
            return
        mode, pan_pos, tilt_pos, voltage = servo_status
        self.servo_status_mode.set(mode)
        self.servo_status_pan.set("-" if pan_pos is None else str(int(pan_pos)))
        self.servo_status_tilt.set("-" if tilt_pos is None else str(int(tilt_pos)))
        if mode == "控制板":
            self.servo_status_voltage.set("-" if voltage is None else f"{int(voltage)} mV")
        else:
            self.servo_status_voltage.set("-")

    def _ensure_servo(self):
        if self.servo is not None:
            return
        settings = self._get_settings()
        self.active_pan_id = settings["pan_id"]
        self.active_tilt_id = settings["tilt_id"]
        if settings.get("servo_mode") == "无刷RS485":
            motor_config_cls, motor_cls = self._get_brushless_motor_classes()
            self.servo = BrushlessDualServoAdapter(
                motor_config_cls=motor_config_cls,
                motor_cls=motor_cls,
                pan_id=self.active_pan_id,
                tilt_id=self.active_tilt_id,
                pan_dev=settings["brushless_pan_dev"],
                tilt_dev=settings["brushless_tilt_dev"],
                pan_baudrate=settings["brushless_pan_baudrate"],
                tilt_baudrate=settings["brushless_tilt_baudrate"],
                pan_txden=settings["brushless_pan_txden"],
                tilt_txden=settings["brushless_tilt_txden"],
                pan_direction_sign=settings["brushless_pan_direction_sign"],
                tilt_direction_sign=settings["brushless_tilt_direction_sign"],
                pan_speed_dps=settings["brushless_pan_speed_dps"],
                tilt_speed_dps=settings["brushless_tilt_speed_dps"],
                pan_min_deg=settings["pan_min"],
                pan_max_deg=settings["pan_max"],
                tilt_min_deg=settings["tilt_min"],
                tilt_max_deg=settings["tilt_max"],
            )
            self.servo.set_angles([(self.active_pan_id, 0.0), (self.active_tilt_id, 0.0)])
            self.servo.move_angle(wait=False)
            self.current_pan_angle = 0.0
            self.current_tilt_angle = 0.0
            self.hw_pan_min.set(float(settings["pan_min"]))
            self.hw_pan_max.set(float(settings["pan_max"]))
            self.hw_tilt_min.set(float(settings["tilt_min"]))
            self.hw_tilt_max.set(float(settings["tilt_max"]))
            self.status_text.set("无刷RS485已连接")
            self.servo_status_mode.set(settings.get("servo_mode"))
            return
        if settings.get("servo_mode") == "控制板":
            self._load_board_modules()
            transport = self._board_transport_cls(
                port=settings["port"],
                baudrate=settings["baudrate"],
                timeout=1.0,
                debug=False,
            )
            driver = self._board_driver_cls(transport)
            self.servo = BoardServoAdapter(
                driver=driver,
                servo_ids=[self.active_pan_id, self.active_tilt_id],
                moving_time=settings["move_time_ms"],
            )
            self.servo.set_angles([(self.active_pan_id, 0.0), (self.active_tilt_id, 0.0)])
            self.servo.move_angle(wait=False)
            self.current_pan_angle = 0.0
            self.current_tilt_angle = 0.0
            self.hw_pan_min.set(-90.0)
            self.hw_pan_max.set(90.0)
            self.hw_tilt_min.set(-90.0)
            self.hw_tilt_max.set(90.0)
            self.status_text.set("控制板不支持硬件边界读取，已设为±90°")
            self.servo_status_mode.set(settings.get("servo_mode"))
            return
        bus_servo_cls = self._get_bus_servo_cls()
        self.servo = bus_servo_cls(
            port=settings["port"],
            baudrate=settings["baudrate"],
            servo_num=2,
            servo_ids=[self.active_pan_id, self.active_tilt_id],
            moving_time=settings["move_time_ms"],
        )
        self.servo.set_angles([(self.active_pan_id, 0.0), (self.active_tilt_id, 0.0)])
        self.servo.move_angle(wait=False)
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0

        max_attempts = 10
        default_min, default_max = -90.0, 90.0

        def normalize_limits(raw_min, raw_max):
            try:
                a = float(raw_min)
                b = float(raw_max)
            except Exception:
                return default_min, default_max, False
            if not (math.isfinite(a) and math.isfinite(b)):
                return default_min, default_max, False
            if a > b:
                a, b = b, a
            a = max(-120.0, min(120.0, a))
            b = max(-120.0, min(120.0, b))
            return a, b, True

        print("[INFO] 正在读取水平舵机硬件边界...")
        for attempt in range(1, max_attempts + 1):
            try:
                pan_min, pan_max = self.servo.read_hardware_angle_limits(self.active_pan_id)
                pan_min, pan_max, ok = normalize_limits(pan_min, pan_max)
                self.root.after(0, lambda: self.hw_pan_min.set(pan_min))
                self.root.after(0, lambda: self.hw_pan_max.set(pan_max))
                if ok:
                    print(f"[INFO] 水平舵机边界读取成功: {pan_min} ~ {pan_max}")
                else:
                    self.status_text.set("水平舵机硬件边界异常，已回退到±90°")
                break
            except Exception as e:
                if attempt < max_attempts:
                    print(f"[WARNING] 读取水平舵机边界失败 ({attempt}/{max_attempts}): {e}。1秒后重试...")
                    time.sleep(1.0)
                else:
                    print(f"[ERROR] 连续 {max_attempts} 次读取水平舵机边界失败，放弃读取。将使用默认软限位。")

        print("[INFO] 正在读取俯仰舵机硬件边界...")
        for attempt in range(1, max_attempts + 1):
            try:
                tilt_min, tilt_max = self.servo.read_hardware_angle_limits(self.active_tilt_id)
                tilt_min, tilt_max, ok = normalize_limits(tilt_min, tilt_max)
                self.root.after(0, lambda: self.hw_tilt_min.set(tilt_min))
                self.root.after(0, lambda: self.hw_tilt_max.set(tilt_max))
                if ok:
                    print(f"[INFO] 俯仰舵机边界读取成功: {tilt_min} ~ {tilt_max}")
                else:
                    self.status_text.set("俯仰舵机硬件边界异常，已回退到±90°")
                break
            except Exception as e:
                if attempt < max_attempts:
                    print(f"[WARNING] 读取俯仰舵机边界失败 ({attempt}/{max_attempts}): {e}。1秒后重试...")
                    time.sleep(1.0)
                else:
                    print(f"[ERROR] 连续 {max_attempts} 次读取俯仰舵机边界失败，放弃读取。将使用默认软限位。")

    def _detect_circle(self, frame_rgb, *, ksize, min_dist, param1, param2, min_radius, max_radius, roi=None):
        offset_x = 0
        offset_y = 0
        if roi is not None:
            x0, y0, x1, y1 = roi
            if x1 - x0 >= 20 and y1 - y0 >= 20:
                frame_rgb = frame_rgb[y0:y1, x0:x1]
                offset_x = x0
                offset_y = y0
        h, w = frame_rgb.shape[:2]
        scale = 0.5
        small_w = max(1, int(w * scale))
        small_h = max(1, int(h * scale))
        frame_small = cv2.resize(frame_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
        # 注意: OpenCV 的 frame_rgb 在这里其实是 BGR 格式
        # frame_small[:, :, 0] = B (Blue)
        # frame_small[:, :, 1] = G (Green)
        # frame_small[:, :, 2] = R (Red)
        green = frame_small[:, :, 1]
        red = frame_small[:, :, 2] # 提取红色通道用于找激光光斑
        
        ksize = int(ksize)
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        blurred_green = cv2.GaussianBlur(green, (ksize, ksize), 1)
        blurred_red = cv2.GaussianBlur(red, (ksize, ksize), 1)
        
        min_dist = max(1, int(int(min_dist) * scale))
        min_radius = max(0, int(int(min_radius) * scale))
        max_radius = max(0, int(int(max_radius) * scale))
        circles = cv2.HoughCircles(
            blurred_green,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=int(param1),
            param2=int(param2),
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return None, blurred_green, blurred_red, offset_x, offset_y, scale
        circles = np.round(circles[0]).astype(int)
        chosen = max(circles, key=lambda c: c[2])
        x = int(round(chosen[0] / scale)) + offset_x
        y = int(round(chosen[1] / scale)) + offset_y
        r = int(round(chosen[2] / scale))
        return (x, y, r), blurred_green, blurred_red, offset_x, offset_y, scale

    def _draw_overlay(self, frame, center, detection, target, pred, radius, error, dt, bounds=None, laser_spot=None):
        # 确保所有坐标都是整数，避免 OpenCV 报 "can't parse center" 错误
        center = (int(round(center[0])), int(round(center[1])))
        target = (int(round(target[0])), int(round(target[1])))
        pred = (int(round(pred[0])), int(round(pred[1])))
        radius = int(round(radius))

        cv2.circle(frame, center, 3, (255, 0, 255), -1)
        cv2.circle(frame, target, 4, (0, 255, 0), -1)
        cv2.circle(frame, pred, 3, (255, 255, 0), -1)
        if detection is not None:
            x, y, r = detection
            # EMA平滑后可能是浮点数，需要转为整型
            x, y, r = int(round(x)), int(round(y)), int(round(r))
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), r, (0, 0, 255), 1)
            if radius > 0:
                cv2.circle(frame, target, radius, (0, 255, 0), 1)
                
        if laser_spot is not None:
            lx, ly = int(round(laser_spot[0])), int(round(laser_spot[1]))
            # 在原图上也画一个黄色的十字星表示激光点
            cv2.line(frame, (lx-10, ly), (lx+10, ly), (255, 255, 0), 2)
            cv2.line(frame, (lx, ly-10), (lx, ly+10), (255, 255, 0), 2)
        error_x, error_y = error
        cv2.putText(frame, f"error_x={error_x:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"error_y={error_y:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"dt={dt:.3f}s", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # 边界到达提示
        if bounds is not None:
            pan_at_min, pan_at_max, tilt_at_min, tilt_at_max = bounds
            h, w = frame.shape[:2]
            # 水平边界提示（左右边框变红）
            if pan_at_min:
                cv2.line(frame, (0, 0), (0, h), (0, 0, 255), 4)
                cv2.putText(frame, "LEFT LIMIT", (10, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if pan_at_max:
                cv2.line(frame, (w-1, 0), (w-1, h), (0, 0, 255), 4)
                cv2.putText(frame, "RIGHT LIMIT", (w-120, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            # 俯仰边界提示（上下边框变红）
            if tilt_at_min:
                cv2.line(frame, (0, 0), (w, 0), (0, 0, 255), 4)
                cv2.putText(frame, "UP LIMIT", (w//2-50, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            if tilt_at_max:
                cv2.line(frame, (0, h-1), (w, h-1), (0, 0, 255), 4)
                cv2.putText(frame, "DOWN LIMIT", (w//2-60, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    def on_close(self):
        self.tracking_active = False
        self._save_settings()
        self.stop_event.set()
        self.detect_stop_event.set()
        
        # 退出 GUI 前强制回正舵机
        if self.servo is not None:
            try:
                print("[INFO] Exiting GUI. Centering servos before shutdown...")
                self._center_servos()
                time.sleep(0.5) # 给舵机一点时间移动到位
            except Exception:
                pass
                
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=2.0)
            self.worker_thread = None
        if self.detect_thread is not None:
            self.detect_thread.join(timeout=2.0)
            self.detect_thread = None
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            try:
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None
        if self.servo is not None:
            try:
                self.servo.cleanup()
            except Exception:
                pass
            self.servo = None
        if self.laser_ranger is not None:
            try:
                self.laser_ranger.stop()
            except Exception:
                pass
            self.laser_ranger = None
        self._close_imu()
        self.root.destroy()

    def _jog(self, delta_pan, delta_tilt):
        if self.servo is None:
            try:
                self._ensure_servo()
            except Exception as exc:
                self.worker_error = str(exc)
                self.status_text.set(f"舵机错误: {exc}")
                return
        self.current_pan_angle = float(self.current_pan_angle) + float(delta_pan)
        self.current_tilt_angle = float(self.current_tilt_angle) + float(delta_tilt)
        self.servo.set_angles(
            [
                (self.active_pan_id, self.current_pan_angle),
                (self.active_tilt_id, self.current_tilt_angle),
            ]
        )
        self.servo.move_angle(wait=False)


def main():
    try:
        from ttkthemes import ThemedTk
        root = ThemedTk(theme="arc") # "arc" 主题的复选框带有清晰的蓝色勾选标记
    except ImportError:
        root = tk.Tk()
        # 如果没有安装 ttkthemes，则回退到 'alt' 主题，它的复选框显示为正常的打勾
        try:
            ttk.Style().theme_use('alt')
        except:
            pass

    root.title("Raspi Optic Fine Align")
    # 允许用户根据需要自适应调整
    root.resizable(True, True)
    app = CircleTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
