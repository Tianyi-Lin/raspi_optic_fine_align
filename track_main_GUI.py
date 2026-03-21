import time
import os
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
        self.root.title("Circle Tracker GUI")
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
        self.move_time_ms = tk.IntVar(value=50)
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
        self.exposure_value = tk.DoubleVar(value=0.0)
        self.analogue_gain = tk.DoubleVar(value=8.0)
        self.ae_enable = tk.BooleanVar(value=True)
        self.ksize = tk.IntVar(value=5)
        self.min_dist = tk.IntVar(value=80)
        self.param1 = tk.IntVar(value=220)
        self.param2 = tk.IntVar(value=35)
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=120)
        self.x_bias = tk.IntVar(value=0)
        self.y_bias = tk.IntVar(value=0)
        self.status_text = tk.StringVar(value="Ready")
        self.latest_frame = None
        self.latest_frame_id = 0
        self.latest_detection = None
        self.latest_detection_time = 0.0
        self.detect_stale_sec = 0.3

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _e: self.on_close())
        self.root.bind("q", lambda _e: self.on_close())
        self._update_settings_from_vars()
        self.root.after(10, self._start_runtime)

    def _update_settings_from_vars(self):
        with self.settings_lock:
            self.settings = {
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
            }

    def _get_settings(self):
        with self.settings_lock:
            return dict(self.settings)

    def _build_ui(self):
        main = ttk.Frame(self.root, padding=8)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main, width=420)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

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
        notebook.add(tab_basic, text="Basic")
        notebook.add(tab_pid, text="PID")
        notebook.add(tab_vision, text="Vision")
        notebook.add(tab_camera, text="Camera")

        tab_basic.columnconfigure(1, weight=1)
        tab_basic.columnconfigure(3, weight=1)
        r = 0
        self._grid_entry(tab_basic, r, 0, "串口", self.port, width=18)
        self._grid_entry(tab_basic, r, 2, "Move ms", self.move_time_ms, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "Baud", self.baudrate, width=10)
        r += 1
        self._grid_entry(tab_basic, r, 0, "Pan ID", self.pan_id, width=8)
        self._grid_entry(tab_basic, r, 2, "Tilt ID", self.tilt_id, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "Ctrl ms", self.control_period_ms, width=8)
        self._grid_entry(tab_basic, r, 2, "Jog deg", self.jog_step_deg, width=8)
        r += 1
        ttk.Checkbutton(tab_basic, text="Enable Track", variable=self.track_enabled).grid(row=r, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="Enable Pan", variable=self.pan_enabled).grid(row=r, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="Enable Tilt", variable=self.tilt_enabled).grid(row=r, column=2, sticky="w", pady=(6, 0))
        r += 1
        btns = ttk.Frame(tab_basic)
        btns.grid(row=r, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        btns.columnconfigure(0, weight=1)
        btns.columnconfigure(1, weight=1)
        btns.columnconfigure(2, weight=1)
        btns.columnconfigure(3, weight=1)
        ttk.Button(btns, text="Start", command=self.start).grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Stop", command=self.stop).grid(row=0, column=1, sticky="ew", padx=(6, 0))
        ttk.Button(btns, text="Reset", command=self.reset_axes).grid(row=0, column=2, sticky="ew", padx=(6, 0))
        ttk.Button(btns, text="Exit", command=self.on_close).grid(row=0, column=3, sticky="ew", padx=(6, 0))

        jog = ttk.LabelFrame(tab_basic, text="Jog", padding=8)
        jog.grid(row=r + 1, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        for c in range(3):
            jog.columnconfigure(c, weight=1)
        ttk.Button(jog, text="Up", command=lambda: self._jog(0.0, +self.jog_step_deg.get())).grid(row=0, column=1, sticky="ew")
        ttk.Button(jog, text="Left", command=lambda: self._jog(-self.jog_step_deg.get(), 0.0)).grid(row=1, column=0, sticky="ew", pady=(6, 0))
        ttk.Button(jog, text="Down", command=lambda: self._jog(0.0, -self.jog_step_deg.get())).grid(row=1, column=1, sticky="ew", pady=(6, 0))
        ttk.Button(jog, text="Right", command=lambda: self._jog(+self.jog_step_deg.get(), 0.0)).grid(row=1, column=2, sticky="ew", pady=(6, 0))

        pid_cols = ttk.Frame(tab_pid)
        pid_cols.pack(fill=tk.BOTH, expand=True)
        pid_cols.columnconfigure(0, weight=1)
        pid_cols.columnconfigure(1, weight=1)
        pid_x_frame = ttk.LabelFrame(pid_cols, text="X Axis", padding=8)
        pid_y_frame = ttk.LabelFrame(pid_cols, text="Y Axis", padding=8)
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
        common = ttk.LabelFrame(tab_pid, text="Common", padding=8)
        common.pack(fill=tk.X, pady=(10, 0))
        common.columnconfigure(0, weight=1)
        common.columnconfigure(1, weight=1)
        r2 = 0
        r2 = self._grid_slider(common, r2, 0, "Deadband", self.error_deadband, 0.0, 30.0)
        r2 = self._grid_slider(common, r2, 0, "MaxDeg/s", self.max_delta_deg_per_sec, 1.0, 200.0)

        tab_vision.columnconfigure(0, weight=1)
        tab_vision.columnconfigure(1, weight=1)
        left_vis = ttk.LabelFrame(tab_vision, text="Hough", padding=8)
        right_vis = ttk.LabelFrame(tab_vision, text="Bias/Blur", padding=8)
        left_vis.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right_vis.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        for c in range(2):
            left_vis.columnconfigure(c, weight=1)
            right_vis.columnconfigure(c, weight=1)
        rv = 0
        rv = self._grid_slider(left_vis, rv, 0, "MinDist", self.min_dist, 10, 300)
        rv = self._grid_slider(left_vis, rv, 0, "Param1", self.param1, 50, 500)
        rv = self._grid_slider(left_vis, rv, 0, "Param2", self.param2, 5, 200)
        rv = self._grid_slider(left_vis, rv, 0, "MinRadius", self.min_radius, 1, 300)
        rv = self._grid_slider(left_vis, rv, 0, "MaxRadius", self.max_radius, 1, 300)
        rv2 = 0
        rv2 = self._grid_slider(right_vis, rv2, 0, "Blur ksize", self.ksize, 3, 19)
        rv2 = self._grid_slider(right_vis, rv2, 0, "X Bias", self.x_bias, -200, 200)
        rv2 = self._grid_slider(right_vis, rv2, 0, "Y Bias", self.y_bias, -200, 200)

        tab_camera.columnconfigure(0, weight=1)
        cam = ttk.Frame(tab_camera)
        cam.pack(fill=tk.BOTH, expand=True)
        cam.columnconfigure(0, weight=1)
        rc = 0
        ttk.Checkbutton(cam, text="Auto Exposure", variable=self.ae_enable).grid(row=rc, column=0, sticky="w", pady=(2, 8))
        rc += 1
        rc = self._grid_slider(cam, rc, 0, "Exposure", self.exposure_value, -8.0, 8.0)
        rc = self._grid_slider(cam, rc, 0, "Gain", self.analogue_gain, 1.0, 22.0)

    def _grid_entry(self, parent, row, col, text, var, width=10):
        ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=(2, 2))
        ttk.Entry(parent, textvariable=var, width=width).grid(row=row, column=col + 1, sticky="ew", pady=(2, 2))

    def _grid_slider(self, parent, row, col, text, var, low, high):
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=col, columnspan=2, sticky="ew", pady=(2, 6))
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
            self.status_text.set("Detecting (tracking off)")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"Init failed: {exc}")
            messagebox.showerror("Init failed", str(exc))

    def start(self):
        if not self.running:
            self._start_runtime()
        self.tracking_active = True
        self.pid_x.reset()
        self.pid_y.reset()
        self.kalman = Kalman2D()
        self.status_text.set("Tracking started")

    def stop(self):
        self.tracking_active = False
        if self.running:
            self.status_text.set("Detecting (tracking off)")

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
                self._sync_camera_controls(s["ae_enable"], s["exposure"], s["gain"])

                frame_rgb = self.picam2.capture_array()
                h, w = frame_rgb.shape[:2]
                center_x, center_y = w // 2, h // 2
                with self.detect_lock:
                    self.latest_frame = (frame_rgb, s)
                    self.latest_frame_id += 1
                    detection = self.latest_detection
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
                if do_track and self.servo is None:
                    try:
                        self._ensure_servo()
                    except Exception as exc:
                        self.worker_error = str(exc)
                        self.stop_event.set()
                        break
                if do_track and s["pan_enabled"]:
                    delta_x = self.pid_x.update(error_x, dt=dt)
                    desired_pan = self.current_pan_angle + delta_x
                    step_pan = max(-max_step, min(max_step, desired_pan - self.current_pan_angle))
                    self.current_pan_angle = max(-90.0, min(90.0, self.current_pan_angle + step_pan))

                if do_track and s["tilt_enabled"]:
                    delta_y = self.pid_y.update(error_y, dt=dt)
                    desired_tilt = self.current_tilt_angle + delta_y
                    step_tilt = max(-max_step, min(max_step, desired_tilt - self.current_tilt_angle))
                    self.current_tilt_angle = max(-90.0, min(90.0, self.current_tilt_angle + step_tilt))

                if do_track and (s["pan_enabled"] or s["tilt_enabled"]):
                    self.servo.set_angles(
                        [
                            (self.active_pan_id, self.current_pan_angle),
                            (self.active_tilt_id, self.current_tilt_angle),
                        ]
                    )
                    self.servo.move_angle(wait=False)

                frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                self._draw_overlay(
                    frame=frame_bgr,
                    center=(center_x, center_y),
                    detection=detection,
                    target=(int(round(target_x)), int(round(target_y))),
                    pred=(int(round(pred_x)), int(round(pred_y))),
                    radius=radius,
                    error=(error_x, error_y),
                    dt=dt,
                )
                frame_rgb_show = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

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

                period_ms = max(10, int(s["control_period_ms"]))
                move_ms = max(0, int(s["move_time_ms"]))
                effective_ms = max(period_ms, move_ms)
                elapsed_ms = int((time.time() - loop_start) * 1000)
                sleep_ms = max(0, effective_ms - elapsed_ms)
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
                detection = self._detect_circle(
                    frame_rgb,
                    ksize=s["ksize"],
                    min_dist=s["min_dist"],
                    param1=s["param1"],
                    param2=s["param2"],
                    min_radius=s["min_radius"],
                    max_radius=s["max_radius"],
                )
                with self.detect_lock:
                    self.latest_detection = detection
                    self.latest_detection_time = time.time()
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
            self.status_text.set(f"Worker error: {self.worker_error}")
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
            if dt > 0:
                self.fps = 1.0 / dt
            self.status_text.set(
                f"FPS={self.fps:.1f}  pan={pan:.2f}  tilt={tilt:.2f}  ex={error_x:.1f}  ey={error_y:.1f}  circle={int(circle_found)} r={radius}  track={int(do_track)}"
            )

        self.after_id = self.root.after(30, self._ui_loop)

    def _sync_camera_controls(self, ae_enable, exposure, gain):
        if self.last_ae_enable == ae_enable and self.last_exposure == exposure and self.last_gain == gain:
            return
        self.picam2.set_controls({"AeEnable": bool(ae_enable), "AwbEnable": True})
        self.picam2.set_controls({"ExposureValue": float(exposure), "AnalogueGain": float(gain)})
        self.last_ae_enable = ae_enable
        self.last_exposure = exposure
        self.last_gain = gain

    def _ensure_camera(self):
        if self.picam2 is not None:
            return
        self.picam2 = Picamera2()
        self.picam2.start_preview(Preview.NULL)
        framerate = 30
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

    def _detect_circle(self, frame_rgb, *, ksize, min_dist, param1, param2, min_radius, max_radius):
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
            return None
        circles = np.round(circles[0]).astype(int)
        chosen = max(circles, key=lambda c: c[2])
        x = int(round(chosen[0] / scale))
        y = int(round(chosen[1] / scale))
        r = int(round(chosen[2] / scale))
        return x, y, r

    def _draw_overlay(self, frame, center, detection, target, pred, radius, error, dt):
        cv2.circle(frame, center, 3, (255, 0, 255), -1)
        cv2.circle(frame, target, 4, (0, 255, 0), -1)
        cv2.circle(frame, pred, 3, (255, 255, 0), -1)
        if detection is not None:
            x, y, r = detection
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
            cv2.circle(frame, (x, y), r, (0, 0, 255), 1)
            if radius > 0:
                cv2.circle(frame, target, radius, (0, 255, 0), 1)
        error_x, error_y = error
        cv2.putText(frame, f"error_x={error_x:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"error_y={error_y:.2f}", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"dt={dt:.3f}s", (10, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    def on_close(self):
        self.tracking_active = False
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
                self.status_text.set(f"Servo error: {exc}")
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
    root = tk.Tk()
    root.geometry("1280x760")
    app = CircleTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
