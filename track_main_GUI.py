import time
import os
import json
import threading
import queue
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk
from picamera2 import Picamera2, Preview

from PID import PID
from bus_servo import BusServo


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
        self.active_pan_id = 1
        self.active_tilt_id = 2
        self.jog_step_deg = tk.DoubleVar(value=1.0)

        self.port = tk.StringVar(value="/dev/ttyAMA0")
        self.baudrate = tk.IntVar(value=115200)
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
        self.max_delta_deg_per_sec = tk.DoubleVar(value=30.0)
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
        self.status_text = tk.StringVar(value="就绪")
        # 舵机角度范围配置（角度制）
        self.pan_min = tk.DoubleVar(value=-90.0)
        self.pan_max = tk.DoubleVar(value=90.0)
        self.tilt_min = tk.DoubleVar(value=-90.0)
        self.tilt_max = tk.DoubleVar(value=90.0)
        
        # 卡尔曼滤波参数
        self.kalman_process_noise = tk.DoubleVar(value=0.03)
        self.kalman_measurement_noise = tk.DoubleVar(value=0.4)
        
        # 硬件物理边界（从舵机读取）
        self.hw_pan_min = tk.DoubleVar(value=-90.0)
        self.hw_pan_max = tk.DoubleVar(value=90.0)
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

        # 用于平滑检测结果的EMA（指数移动平均）状态
        self.smoothed_detection = None
        self.ema_alpha = 0.3  # 平滑系数，越小越平滑但延迟越大，越大响应越快但抖动越大

        self._load_settings()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _e: self.on_close())
        self.root.bind("q", lambda _e: self.on_close())
        self._update_settings_from_vars()
        self.root.after(10, self._start_runtime)

    def _update_settings_from_vars(self):
        fallback = {
            "baudrate": 115200,
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
            "max_delta_deg_per_sec": 30.0,
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
            "pan_min": -90.0,
            "pan_max": 90.0,
            "tilt_min": -90.0,
            "tilt_max": 90.0,
            "hw_pan_min": -90.0,
            "hw_pan_max": 90.0,
            "hw_tilt_min": -90.0,
            "hw_tilt_max": 90.0,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.4,
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
                "baudrate": safe_int(self.baudrate, "baudrate"),
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
                "pan_min": safe_float(self.pan_min, "pan_min"),
                "pan_max": safe_float(self.pan_max, "pan_max"),
            "tilt_min": safe_float(self.tilt_min, "tilt_min"),
            "tilt_max": safe_float(self.tilt_max, "tilt_max"),
            "kalman_process_noise": safe_float(self.kalman_process_noise, "kalman_process_noise"),
            "kalman_measurement_noise": safe_float(self.kalman_measurement_noise, "kalman_measurement_noise"),
        }

    def _get_settings(self):
        # 实时从GUI变量读取，确保修改立即生效
        defaults = {
            "port": "/dev/ttyUSB0",
            "baudrate": 115200,
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
            "max_delta_deg_per_sec": 30.0,
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
            "pan_min": -90.0,
            "pan_max": 90.0,
            "tilt_min": -90.0,
            "tilt_max": 90.0,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.4,
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
            "port": str(self.port.get()) if self.port.get() else defaults["port"],
            "baudrate": safe_int(self.baudrate, "baudrate"),
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
            "pan_min": safe_float(self.pan_min, "pan_min"),
            "pan_max": safe_float(self.pan_max, "pan_max"),
            "tilt_min": safe_float(self.tilt_min, "tilt_min"),
            "tilt_max": safe_float(self.tilt_max, "tilt_max"),
            "kalman_process_noise": safe_float(self.kalman_process_noise, "kalman_process_noise"),
            "kalman_measurement_noise": safe_float(self.kalman_measurement_noise, "kalman_measurement_noise"),
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
            "port": self.port,
            "baudrate": self.baudrate,
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
            "pan_min": self.pan_min,
            "pan_max": self.pan_max,
            "tilt_min": self.tilt_min,
            "tilt_max": self.tilt_max,
            "kalman_process_noise": self.kalman_process_noise,
            "kalman_measurement_noise": self.kalman_measurement_noise,
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
                var.set(value)

    def _save_settings(self):
        path = self._settings_path()
        try:
            data = {
                "port": self.port.get(),
                "baudrate": int(self.baudrate.get()),
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
                "pan_min": float(self.pan_min.get()),
                "pan_max": float(self.pan_max.get()),
                "tilt_min": float(self.tilt_min.get()),
                "tilt_max": float(self.tilt_max.get()),
                "kalman_process_noise": float(self.kalman_process_noise.get()),
                "kalman_measurement_noise": float(self.kalman_measurement_noise.get()),
            }
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 设置已保存到: {path}")
        except Exception as e:
            print(f"[ERROR] 保存设置失败: {e}")

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
        self._grid_entry(tab_basic, r, 0, "串口", self.port, width=18)
        self._grid_entry(tab_basic, r, 2, "移动时间ms", self.move_time_ms, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "波特率", self.baudrate, width=10)
        r += 1
        self._grid_entry(tab_basic, r, 0, "水平ID", self.pan_id, width=8)
        self._grid_entry(tab_basic, r, 2, "俯仰ID", self.tilt_id, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "控制周期ms", self.control_period_ms, width=8)
        self._grid_entry(tab_basic, r, 2, "点动角度", self.jog_step_deg, width=8)
        r += 1
        ttk.Checkbutton(tab_basic, text="启用跟踪", variable=self.track_enabled).grid(row=r, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="启用水平", variable=self.pan_enabled).grid(row=r, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="启用俯仰", variable=self.tilt_enabled).grid(row=r, column=2, sticky="w", pady=(6, 0))
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
        ttk.Button(jog, text="左", command=lambda: self._jog(-self.jog_step_deg.get(), 0.0)).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(jog, text="下", command=lambda: self._jog(0.0, -self.jog_step_deg.get())).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(jog, text="右", command=lambda: self._jog(+self.jog_step_deg.get(), 0.0)).grid(row=1, column=2, sticky="ew", pady=(6, 0))

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
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kP", self.kp_x, 0.0, 0.1)
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kI", self.ki_x, 0.0, 0.2)
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kD", self.kd_x, 0.0, 0.02)
        row_y = 0
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kP", self.kp_y, 0.0, 0.1)
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kI", self.ki_y, 0.0, 0.2)
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kD", self.kd_y, 0.0, 0.02)
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
        rk = self._grid_slider(kalman_frame, rk, 0, "过程噪声(运动不可预测性)", self.kalman_process_noise, 0.001, 0.5)
        ttk.Label(kalman_frame, text="越小:平滑但延迟大; 越大:响应快但易抖动", font=("", 8), foreground="gray", wraplength=180).grid(row=rk-1, column=2, sticky="w", padx=(6, 0))
        rk = self._grid_slider(kalman_frame, rk, 0, "测量噪声(检测结果不稳定性)", self.kalman_measurement_noise, 0.01, 5.0)
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
        rv2 = self._grid_slider(right_vis, rv2, 0, "X偏置", self.x_bias, -200, 200)
        rv2 = self._grid_slider(right_vis, rv2, 0, "Y偏置", self.y_bias, -200, 200)

        tab_camera.columnconfigure(0, weight=1)
        cam = ttk.Frame(tab_camera)
        cam.pack(fill=tk.BOTH, expand=True)
        cam.columnconfigure(0, weight=1)
        rc = 0
        rc = self._grid_slider(cam, rc, 0, "相机FPS", self.camera_fps, 10, 120)
        
        # 添加红色警告提示Label (初始隐藏或为空)
        self.fps_warning_var = tk.StringVar(value="")
        self.fps_warning_label = ttk.Label(cam, textvariable=self.fps_warning_var, foreground="red", font=("", 9, "bold"))
        self.fps_warning_label.grid(row=rc, column=0, sticky="w", pady=(0, 4))
        rc += 1
        
        ttk.Checkbutton(cam, text="自动曝光", variable=self.ae_enable).grid(row=rc, column=0, sticky="w", pady=(2, 8))
        rc += 1
        rc = self._grid_slider(cam, rc, 0, "曝光(us)", self.exposure_value, 100, 100000)
        rc = self._grid_slider(cam, rc, 0, "增益", self.analogue_gain, 1.0, 22.0)

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
        step = (high - low) / 100.0  # 每次按键移动1%
        if isinstance(var, tk.IntVar):
            step = max(1, int(step))

        def _on_key(event):
            current = var.get()
            if event.keysym == "Left":
                new_val = max(low, current - step)
            elif event.keysym == "Right":
                new_val = min(high, current + step)
            elif event.keysym == "Home":
                new_val = low
            elif event.keysym == "End":
                new_val = high
            else:
                return
            var.set(new_val)

        scale.bind("<Left>", _on_key)
        scale.bind("<Right>", _on_key)
        scale.bind("<Home>", _on_key)
        scale.bind("<End>", _on_key)
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
        self.status_text.set("跟踪已开始")

    def stop(self):
        self.tracking_active = False
        if self.running:
            self.status_text.set("检测中（未跟踪）")

    def reset_axes(self):
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        self.pid_x.reset()
        self.pid_y.reset()
        self.kalman = Kalman2D()
        with self.detect_lock:
            self.latest_detection = None
            self.latest_detection_time = 0.0
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
                target_x = filtered_x + float(s["x_bias"])
                target_y = filtered_y + float(s["y_bias"])

                if not s["track_enabled"]:
                    target_x = float(center_x)
                    target_y = float(center_y)

                self.pid_x.set_gains(s["kp_x"], s["ki_x"], s["kd_x"])
                self.pid_y.set_gains(s["kp_y"], s["ki_y"], s["kd_y"])
                
                self.kalman.update_params(s["kalman_process_noise"], s["kalman_measurement_noise"])

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
                if do_track:
                    try:
                        # 每次循环都同步角度范围到舵机驱动层（确保GUI修改立即生效）
                        self._ensure_servo()
                    except Exception as exc:
                        self.worker_error = str(exc)
                        self.stop_event.set()
                        break
                # 使用GUI配置和硬件物理边界的交集作为最终限制
                # max(硬件最小, GUI最小) 和 min(硬件最大, GUI最大)
                pan_min = max(float(s.get("hw_pan_min", -90.0)), float(s.get("pan_min", -90.0)))
                pan_max = min(float(s.get("hw_pan_max", 90.0)), float(s.get("pan_max", 90.0)))
                tilt_min = max(float(s.get("hw_tilt_min", -90.0)), float(s.get("tilt_min", -90.0)))
                tilt_max = min(float(s.get("hw_tilt_max", 90.0)), float(s.get("tilt_max", 90.0)))

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

                # 更新后再次检测是否到达边界，用于视觉提示（消除延迟）
                pan_at_min = self.current_pan_angle <= pan_min
                pan_at_max = self.current_pan_angle >= pan_max
                tilt_at_min = self.current_tilt_angle <= tilt_min
                tilt_at_max = self.current_tilt_angle >= tilt_max

                if do_track and (s["pan_enabled"] or s["tilt_enabled"]):
                    self.servo.set_angles(
                        [
                            (self.active_pan_id, self.current_pan_angle),
                            (self.active_tilt_id, self.current_tilt_angle),
                        ]
                    )
                    self.servo.move_angle(wait=False)

                # Picamera2 RGB888 is actually BGR layout [B,G,R]; keep as-is for OpenCV (BGR)
                # and convert to RGB only once for display
                self._draw_overlay(
                    frame=frame_rgb,
                    center=(center_x, center_y),
                    detection=detection,
                    target=(int(round(target_x)), int(round(target_y))),
                    pred=(int(round(pred_x)), int(round(pred_y))),
                    radius=radius,
                    error=(error_x, error_y),
                    dt=dt,
                    bounds=(pan_at_min, pan_at_max, tilt_at_min, tilt_at_max),
                )
                
                # 创建双屏显示：原图 + 绿色通道处理图
                frame_rgb_disp = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2RGB)
                
                if green_data is not None:
                    blurred_green, offset_x, offset_y, scale = green_data
                    
                    # 创建一个全黑的RGB图像用于放绿色通道
                    green_rgb = np.zeros((blurred_green.shape[0], blurred_green.shape[1], 3), dtype=np.uint8)
                    # 将灰度数据放入绿色通道 (RGB顺序，索引为1)
                    green_rgb[:, :, 1] = blurred_green
                    
                    # 如果有ROI，将其放回原尺寸的黑底图像中
                    full_green = np.zeros_like(frame_rgb_disp)
                    gh, gw = green_rgb.shape[:2]
                    
                    # 计算放回原图的位置
                    orig_w = int(gw / scale)
                    orig_h = int(gh / scale)
                    green_resized = cv2.resize(green_rgb, (orig_w, orig_h))
                    
                    # 放入全尺寸图中
                    end_y = min(h, offset_y + orig_h)
                    end_x = min(w, offset_x + orig_w)
                    roi_h = end_y - offset_y
                    roi_w = end_x - offset_x
                    
                    if roi_h > 0 and roi_w > 0:
                        full_green[offset_y:end_y, offset_x:end_x] = green_resized[:roi_h, :roi_w]
                    
                    # 在绿色通道图上标注检测到的圆圈
                    if detection is not None:
                        x, y, r = detection
                        x, y, r = int(round(x)), int(round(y)), int(round(r))
                        cv2.circle(full_green, (x, y), 3, (255, 0, 0), -1)  # 红心
                        cv2.circle(full_green, (x, y), r, (255, 0, 0), 2)   # 红圈
                    
                    # 在图上添加文字说明
                    cv2.putText(full_green, "Green Channel (Processed)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    # 水平拼接
                    frame_rgb_show = np.hstack((frame_rgb_disp, full_green))
                else:
                    # 如果还没有处理好的图，就用黑图补齐
                    full_black = np.zeros_like(frame_rgb_disp)
                    cv2.putText(full_black, "Green Channel (Waiting...)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    frame_rgb_show = np.hstack((frame_rgb_disp, full_black))

                payload = (
                    frame_rgb_show,
                    dt,
                    (error_x, error_y),
                    (self.current_pan_angle, self.current_tilt_angle),
                    circle_found,
                    radius,
                    do_track,
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
                detection, blurred_green, offset_x, offset_y, scale = self._detect_circle(
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
                    self.latest_green_channel = (blurred_green, offset_x, offset_y, scale)
                    last_processed_id = frame_id
        except Exception as exc:
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
            frame_rgb_show, dt, (error_x, error_y), (pan, tilt), circle_found, radius, do_track = latest
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
            self.status_text.set(
                f"相机FPS={self.fps_cam:.1f}  控制Hz={self.fps_ctrl:.1f}  水平={pan:.2f}  俯仰={tilt:.2f}  误差X={error_x:.1f}  误差Y={error_y:.1f}  圆={int(circle_found)}  半径={radius}  跟踪={int(do_track)}"
            )
            
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

    def _ensure_servo(self):
        if self.servo is not None:
            return
        settings = self._get_settings()
        self.active_pan_id = settings["pan_id"]
        self.active_tilt_id = settings["tilt_id"]
        self.servo = BusServo(
            port=settings["port"],
            baudrate=settings["baudrate"],
            servo_num=2,
            servo_ids=[self.active_pan_id, self.active_tilt_id],
            moving_time=settings["move_time_ms"],
        )
        # 程序启动时，默认让两个舵机回到中间(0度)
        self.servo.set_angles([(self.active_pan_id, 0.0), (self.active_tilt_id, 0.0)])
        self.servo.move_angle(wait=False)
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        
        # 读取舵机硬件的物理边界并更新到GUI
        pan_min, pan_max = self.servo.read_hardware_angle_limits(self.active_pan_id)
        tilt_min, tilt_max = self.servo.read_hardware_angle_limits(self.active_tilt_id)
        
        # 使用after以确保在主线程更新GUI
        self.root.after(0, lambda: self.hw_pan_min.set(pan_min))
        self.root.after(0, lambda: self.hw_pan_max.set(pan_max))
        self.root.after(0, lambda: self.hw_tilt_min.set(tilt_min))
        self.root.after(0, lambda: self.hw_tilt_max.set(tilt_max))

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
        green = frame_small[:, :, 1]
        ksize = int(ksize)
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(green, (ksize, ksize), 1)
        min_dist = max(1, int(int(min_dist) * scale))
        min_radius = max(0, int(int(min_radius) * scale))
        max_radius = max(0, int(int(max_radius) * scale))
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=min_dist,
            param1=int(param1),
            param2=int(param2),
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None:
            return None, blurred, offset_x, offset_y, scale
        circles = np.round(circles[0]).astype(int)
        chosen = max(circles, key=lambda c: c[2])
        x = int(round(chosen[0] / scale)) + offset_x
        y = int(round(chosen[1] / scale)) + offset_y
        r = int(round(chosen[2] / scale))
        return (x, y, r), blurred, offset_x, offset_y, scale

    def _draw_overlay(self, frame, center, detection, target, pred, radius, error, dt, bounds=None):
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
