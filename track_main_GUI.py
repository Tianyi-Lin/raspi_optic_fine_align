import time
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

    def reset(self, x, y):
        self.kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.initialized = True

    def update(self, measurement):
        if measurement is not None and not self.initialized:
            self.reset(measurement[0], measurement[1])
        prediction = self.kf.predict()
        if measurement is not None:
            measured = np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
            estimate = self.kf.correct(measured)
        else:
            estimate = prediction
        return float(estimate[0]), float(estimate[1]), float(prediction[0]), float(prediction[1])


class CircleTrackerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Circle Tracker GUI")
        self.running = False
        self.picam2 = None
        self.servo = None
        self.worker_thread = None
        self.stop_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.settings_lock = threading.Lock()
        self.settings = {}
        self.last_exposure = None
        self.last_gain = None
        self.fps = 0.0
        self.after_id = None
        self.kalman = Kalman2D()
        self.pid_x = PID(kP=0.0075, kI=0.025, kD=0.000005, output_bound_low=-12, output_bound_high=12)
        self.pid_y = PID(kP=0.01, kI=0.02, kD=0.000005, output_bound_low=-12, output_bound_high=12)
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        self.active_pan_id = 1
        self.active_tilt_id = 2

        self.port = tk.StringVar(value="/dev/ttyAMA0")
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
        self.ksize = tk.IntVar(value=5)
        self.min_dist = tk.IntVar(value=80)
        self.param1 = tk.IntVar(value=220)
        self.param2 = tk.IntVar(value=35)
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=120)
        self.x_bias = tk.IntVar(value=0)
        self.y_bias = tk.IntVar(value=0)
        self.status_text = tk.StringVar(value="Ready")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _e: self.on_close())
        self.root.bind("q", lambda _e: self.on_close())
        self._update_settings_from_vars()

    def _update_settings_from_vars(self):
        with self.settings_lock:
            self.settings = {
                "port": self.port.get(),
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

        left = ttk.Frame(main)
        left.pack(side=tk.LEFT, fill=tk.Y)

        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        self.preview_label = ttk.Label(right)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right, textvariable=self.status_text).pack(anchor=tk.W, pady=(6, 0))

        ttk.Label(left, text="串口").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.port, width=16).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(left, text="Pan ID").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.pan_id, width=8).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(left, text="Tilt ID").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.tilt_id, width=8).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(left, text="Move Time ms").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.move_time_ms, width=8).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(left, text="Control Period ms").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.control_period_ms, width=8).pack(anchor=tk.W, pady=(0, 6))
        ttk.Checkbutton(left, text="Enable Track", variable=self.track_enabled).pack(anchor=tk.W, pady=(0, 6))
        ttk.Checkbutton(left, text="Enable Pan", variable=self.pan_enabled).pack(anchor=tk.W, pady=(0, 6))
        ttk.Checkbutton(left, text="Enable Tilt", variable=self.tilt_enabled).pack(anchor=tk.W, pady=(0, 6))
        ttk.Button(left, text="Start", command=self.start).pack(fill=tk.X)
        ttk.Button(left, text="Stop", command=self.stop).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(left, text="Reset", command=self.reset_axes).pack(fill=tk.X, pady=(4, 10))
        ttk.Button(left, text="Exit", command=self.on_close).pack(fill=tk.X, pady=(0, 10))

        self._slider(left, "kP X", self.kp_x, 0.0, 0.1, 0.0001)
        self._slider(left, "kI X", self.ki_x, 0.0, 0.2, 0.0001)
        self._slider(left, "kD X", self.kd_x, 0.0, 0.02, 0.000001)
        self._slider(left, "kP Y", self.kp_y, 0.0, 0.1, 0.0001)
        self._slider(left, "kI Y", self.ki_y, 0.0, 0.2, 0.0001)
        self._slider(left, "kD Y", self.kd_y, 0.0, 0.02, 0.000001)
        self._slider(left, "Deadband", self.error_deadband, 0.0, 30.0, 0.1)
        self._slider(left, "MaxDeg/s", self.max_delta_deg_per_sec, 1.0, 200.0, 1.0)
        self._slider(left, "Exposure", self.exposure_value, -8.0, 8.0, 0.1)
        self._slider(left, "Gain", self.analogue_gain, 1.0, 22.0, 0.1)
        self._slider(left, "Blur ksize", self.ksize, 3, 19, 2)
        self._slider(left, "MinDist", self.min_dist, 10, 300, 1)
        self._slider(left, "Param1", self.param1, 50, 500, 1)
        self._slider(left, "Param2", self.param2, 5, 200, 1)
        self._slider(left, "MinRadius", self.min_radius, 1, 300, 1)
        self._slider(left, "MaxRadius", self.max_radius, 1, 300, 1)
        self._slider(left, "X Bias", self.x_bias, -200, 200, 1)
        self._slider(left, "Y Bias", self.y_bias, -200, 200, 1)

    def _slider(self, parent, text, var, low, high, step):
        ttk.Label(parent, text=text).pack(anchor=tk.W)
        scale = ttk.Scale(parent, from_=low, to=high, variable=var)
        scale.pack(fill=tk.X, pady=(0, 4))
        if isinstance(var, tk.IntVar):
            ttk.Label(parent, textvariable=var).pack(anchor=tk.E, pady=(0, 4))
        else:
            label_var = tk.StringVar(value=f"{var.get():.4f}")
            ttk.Label(parent, textvariable=label_var).pack(anchor=tk.E, pady=(0, 4))

            def _on_change(*_):
                label_var.set(f"{var.get():.4f}")

            var.trace_add("write", _on_change)

    def start(self):
        if self.running:
            return
        try:
            self.stop_event.clear()
            self._update_settings_from_vars()
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
            time.sleep(0.3)
            self.picam2.set_controls({"AeEnable": False, "AwbEnable": True})
            self.last_exposure = None
            self.last_gain = None
            settings = self._get_settings()
            self.active_pan_id = settings["pan_id"]
            self.active_tilt_id = settings["tilt_id"]
            self.servo = BusServo(
                port=settings["port"],
                baudrate=9600,
                servo_num=2,
                servo_ids=[self.active_pan_id, self.active_tilt_id],
                moving_time=settings["move_time_ms"],
            )
            self.pid_x.reset()
            self.pid_y.reset()
            self.kalman = Kalman2D()
            self.current_pan_angle = 0.0
            self.current_tilt_angle = 0.0
            self.running = True
            self.status_text.set("Tracking started")
            self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self.worker_thread.start()
            self.after_id = self.root.after(10, self._ui_loop)
        except Exception as exc:
            self.status_text.set(f"Start failed: {exc}")
            messagebox.showerror("Start failed", str(exc))
            self.stop()

    def stop(self):
        self.running = False
        self.stop_event.set()
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.worker_thread is not None:
            self.worker_thread.join(timeout=1.0)
            self.worker_thread = None
        if self.picam2 is not None:
            try:
                self.picam2.stop()
            except Exception:
                pass
            self.picam2 = None
        if self.servo is not None:
            try:
                self.servo.cleanup()
            except Exception:
                pass
            self.servo = None
        self.status_text.set("Stopped")

    def reset_axes(self):
        self.current_pan_angle = 0.0
        self.current_tilt_angle = 0.0
        self.pid_x.reset()
        self.pid_y.reset()
        if self.servo is not None:
            self.servo.set_angles(
                [
                    (self.active_pan_id, 0.0),
                    (self.active_tilt_id, 0.0),
                ]
            )
            self.servo.move_angle(wait=False)

    def _worker_loop(self):
        last_time = time.time()
        while not self.stop_event.is_set():
            loop_start = time.time()
            dt = max(loop_start - last_time, 1e-4)
            last_time = loop_start

            if self.picam2 is None or self.servo is None:
                time.sleep(0.01)
                continue

            s = self._get_settings()
            self.servo.moving_time = max(0, int(s["move_time_ms"]))
            self._sync_camera_controls(s["exposure"], s["gain"])

            frame_rgb = self.picam2.capture_array()
            h, w = frame_rgb.shape[:2]
            center_x, center_y = w // 2, h // 2
            detection = self._detect_circle(
                frame_rgb,
                ksize=s["ksize"],
                min_dist=s["min_dist"],
                param1=s["param1"],
                param2=s["param2"],
                min_radius=s["min_radius"],
                max_radius=s["max_radius"],
            )

            measurement = None
            radius = 0
            if detection is not None:
                measurement = (float(detection[0]), float(detection[1]))
                radius = int(detection[2])

            filtered_x, filtered_y, pred_x, pred_y = self.kalman.update(measurement)
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

            max_step = float(s["max_delta_deg_per_sec"]) * dt

            if s["pan_enabled"]:
                delta_x = self.pid_x.update(error_x, dt=dt)
                desired_pan = self.current_pan_angle + delta_x
                step_pan = max(-max_step, min(max_step, desired_pan - self.current_pan_angle))
                self.current_pan_angle = max(-90.0, min(90.0, self.current_pan_angle + step_pan))

            if s["tilt_enabled"]:
                delta_y = self.pid_y.update(error_y, dt=dt)
                desired_tilt = self.current_tilt_angle + delta_y
                step_tilt = max(-max_step, min(max_step, desired_tilt - self.current_tilt_angle))
                self.current_tilt_angle = max(-90.0, min(90.0, self.current_tilt_angle + step_tilt))

            if s["pan_enabled"] or s["tilt_enabled"]:
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

    def _ui_loop(self):
        if not self.running:
            return
        self._update_settings_from_vars()
        latest = None
        try:
            while True:
                latest = self.frame_queue.get_nowait()
        except Exception:
            pass

        if latest is not None:
            frame_rgb_show, dt, (error_x, error_y), (pan, tilt) = latest
            image = Image.fromarray(frame_rgb_show)
            photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=photo)
            self.preview_label.image = photo
            if dt > 0:
                self.fps = 1.0 / dt
            self.status_text.set(
                f"FPS={self.fps:.1f}  pan={pan:.2f}  tilt={tilt:.2f}  ex={error_x:.1f}  ey={error_y:.1f}"
            )

        self.after_id = self.root.after(30, self._ui_loop)

    def _sync_camera_controls(self, exposure, gain):
        if self.last_exposure == exposure and self.last_gain == gain:
            return
        self.picam2.set_controls({"ExposureValue": exposure, "AnalogueGain": gain})
        self.last_exposure = exposure
        self.last_gain = gain

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
        self.stop()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1280x760")
    app = CircleTrackerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
