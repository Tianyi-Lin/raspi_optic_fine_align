import time
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
        self.last_loop_time = time.time()
        self.current_pan_angle = 0.0
        self.last_exposure = None
        self.last_gain = None
        self.fps = 0.0
        self.kalman = Kalman2D()
        self.pid = PID(kP=0.0075, kI=0.025, kD=0.000005, output_bound_low=-12, output_bound_high=12)

        self.port = tk.StringVar(value="/dev/ttyAMA0")
        self.motor_id = tk.IntVar(value=1)
        self.max_speed = tk.IntVar(value=50)
        self.track_enabled = tk.BooleanVar(value=True)
        self.kp = tk.DoubleVar(value=0.0075)
        self.ki = tk.DoubleVar(value=0.025)
        self.kd = tk.DoubleVar(value=0.000005)
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
        ttk.Label(left, text="Motor ID").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.motor_id, width=8).pack(anchor=tk.W, pady=(0, 6))
        ttk.Label(left, text="Move Time ms").pack(anchor=tk.W)
        ttk.Entry(left, textvariable=self.max_speed, width=8).pack(anchor=tk.W, pady=(0, 6))
        ttk.Checkbutton(left, text="Enable Track", variable=self.track_enabled).pack(anchor=tk.W, pady=(0, 6))
        ttk.Button(left, text="Start", command=self.start).pack(fill=tk.X)
        ttk.Button(left, text="Stop", command=self.stop).pack(fill=tk.X, pady=(4, 0))
        ttk.Button(left, text="Reset Pan", command=self.reset_pan).pack(fill=tk.X, pady=(4, 10))

        self._slider(left, "kP", self.kp, 0.0, 0.1, 0.0001)
        self._slider(left, "kI", self.ki, 0.0, 0.2, 0.0001)
        self._slider(left, "kD", self.kd, 0.0, 0.02, 0.000001)
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
            self.servo = BusServo(
                port=self.port.get(),
                baudrate=9600,
                servo_num=1,
                servo_ids=[int(self.motor_id.get())],
                moving_time=int(self.max_speed.get()),
            )
            self.pid.reset()
            self.kalman = Kalman2D()
            self.current_pan_angle = 0.0
            self.last_loop_time = time.time()
            self.running = True
            self.status_text.set("Tracking started")
            self.root.after(1, self._loop)
        except Exception as exc:
            self.status_text.set(f"Start failed: {exc}")
            messagebox.showerror("Start failed", str(exc))
            self.stop()

    def stop(self):
        self.running = False
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

    def reset_pan(self):
        self.current_pan_angle = 0.0
        self.pid.reset()
        if self.servo is not None:
            self.servo.set_angle(int(self.motor_id.get()), 0.0)
            self.servo.move_angle()

    def _loop(self):
        if not self.running:
            return
        loop_start = time.time()
        now = loop_start
        dt = max(now - self.last_loop_time, 1e-4)
        self.last_loop_time = now
        self.pid.set_gains(self.kp.get(), self.ki.get(), self.kd.get())

        if self.picam2 is None or self.servo is None:
            self.stop()
            return

        self._sync_camera_controls()

        frame = self.picam2.capture_array()
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        detection = self._detect_circle(frame)

        measurement = None
        radius = 0
        if detection is not None:
            measurement = (float(detection[0]), float(detection[1]))
            radius = int(detection[2])

        filtered_x, filtered_y, pred_x, pred_y = self.kalman.update(measurement)
        target_x = filtered_x + self.x_bias.get()
        target_y = filtered_y + self.y_bias.get()

        if not self.track_enabled.get():
            target_x = float(center_x)
            target_y = float(center_y)

        error_x = float(center_x) - target_x
        if abs(error_x) < float(self.error_deadband.get()):
            error_x = 0.0

        delta_output = self.pid.update(error_x, dt=dt)
        desired_angle = self.current_pan_angle + delta_output
        max_step = float(self.max_delta_deg_per_sec.get()) * dt
        step = max(-max_step, min(max_step, desired_angle - self.current_pan_angle))
        self.current_pan_angle = max(-90.0, min(90.0, self.current_pan_angle + step))
        self.servo.set_angle(int(self.motor_id.get()), self.current_pan_angle)
        self.servo.move_angle()

        self._draw_overlay(
            frame=frame,
            center=(center_x, center_y),
            detection=detection,
            target=(int(round(target_x)), int(round(target_y))),
            pred=(int(round(pred_x)), int(round(pred_y))),
            radius=radius,
            error=error_x,
            dt=dt,
        )

        image = Image.fromarray(frame)
        photo = ImageTk.PhotoImage(image=image)
        self.preview_label.configure(image=photo)
        self.preview_label.image = photo

        elapsed = max(time.time() - loop_start, 1e-6)
        self.fps = 1.0 / elapsed
        self.status_text.set(
            f"FPS={self.fps:.1f}  pan={self.current_pan_angle:.2f}  error={error_x:.2f}  running={self.running}"
        )
        self.root.after(1, self._loop)

    def _sync_camera_controls(self):
        exposure = float(self.exposure_value.get())
        gain = float(self.analogue_gain.get())
        if self.last_exposure == exposure and self.last_gain == gain:
            return
        self.picam2.set_controls({"ExposureValue": exposure, "AnalogueGain": gain})
        self.last_exposure = exposure
        self.last_gain = gain

    def _detect_circle(self, frame):
        green = frame[:, :, 1]
        ksize = int(self.ksize.get())
        if ksize < 3:
            ksize = 3
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(green, (ksize, ksize), 1)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=int(self.min_dist.get()),
            param1=int(self.param1.get()),
            param2=int(self.param2.get()),
            minRadius=int(self.min_radius.get()),
            maxRadius=int(self.max_radius.get()),
        )
        if circles is None:
            return None
        circles = np.round(circles[0]).astype(int)
        chosen = max(circles, key=lambda c: c[2])
        return int(chosen[0]), int(chosen[1]), int(chosen[2])

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
        cv2.putText(frame, f"error_x={error:.2f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        cv2.putText(frame, f"dt={dt:.3f}s", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

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
