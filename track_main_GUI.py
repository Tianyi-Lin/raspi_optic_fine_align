import time
import os
import json
import math
import threading
import queue
import multiprocessing as mp
import sys
import signal
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


def _parse_core_spec(spec):
    cores = set()
    text = str(spec).strip()
    if not text:
        return cores
    for part in text.split(","):
        p = part.strip()
        if not p:
            continue
        if "-" in p:
            a, b = p.split("-", 1)
            start = max(0, int(a))
            end = max(0, int(b))
            if end < start:
                start, end = end, start
            cores.update(range(start, end + 1))
        else:
            cores.add(max(0, int(p)))
    return cores


def _set_current_process_affinity(core_spec):
    try:
        if os.name != "posix":
            return False
        if not hasattr(os, "sched_setaffinity"):
            return False
        cores = _parse_core_spec(core_spec)
        if not cores:
            return False
        os.sched_setaffinity(0, cores)
        return True
    except Exception:
        return False


def _detect_circle_np(frame_rgb, *, ksize, min_dist, param1, param2, min_radius, max_radius):
    h, w = frame_rgb.shape[:2]
    scale = 0.5
    small_w = max(1, int(w * scale))
    small_h = max(1, int(h * scale))
    frame_small = cv2.resize(frame_rgb, (small_w, small_h), interpolation=cv2.INTER_AREA)
    green = frame_small[:, :, 1]
    k = int(ksize)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    blurred_green = cv2.GaussianBlur(green, (k, k), 1)
    min_dist_s = max(1, int(int(min_dist) * scale))
    min_radius_s = max(0, int(int(min_radius) * scale))
    max_radius_s = max(0, int(int(max_radius) * scale))
    circles = cv2.HoughCircles(
        blurred_green,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=min_dist_s,
        param1=int(param1),
        param2=int(param2),
        minRadius=min_radius_s,
        maxRadius=max_radius_s,
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    chosen = max(circles, key=lambda c: c[2])
    x = int(round(chosen[0] / scale))
    y = int(round(chosen[1] / scale))
    r = int(round(chosen[2] / scale))
    return x, y, r


def _center_crop_rect(full_w: int, full_h: int, ratio: float):
    try:
        r = float(ratio)
    except Exception:
        r = 1.0
    r = max(0.05, min(1.0, r))
    w = max(1, int(full_w))
    h = max(1, int(full_h))
    crop_w = max(32, int(round(w * r)))
    crop_h = max(32, int(round(h * r)))
    crop_w = min(crop_w, w)
    crop_h = min(crop_h, h)
    crop_w = max(2, (crop_w // 2) * 2)
    crop_h = max(2, (crop_h // 2) * 2)
    x0 = max(0, (w - crop_w) // 2)
    y0 = max(0, (h - crop_h) // 2)
    return x0, y0, crop_w, crop_h


def _vision_process_main(stop_event, settings_queue, status_queue, latest_det):
    _set_current_process_affinity("1,2")
    settings = {
        "core_vision": "1,2",
        "camera_raw_width": 640,
        "camera_raw_height": 640,
        "camera_fps": 120,
        "multiprocess_preview": True,
        "sensor_bit_depth": 10,
        "video_crop_ratio": 1.0,
        "preview_push_hz": 15,
        "show_debug_panels": False,
        "laser_align_mode": False,
        "laser_threshold": 180,
        "hough_crop_width": 640,
        "hough_crop_height": 640,
        "mp_roi_enabled": True,
        "mp_roi_stale_sec": 0.25,
        "ksize": 5,
        "min_dist": 80,
        "param1": 220,
        "param2": 35,
        "min_radius": 20,
        "max_radius": 120,
    }
    picam2 = None
    current_cfg = None
    last_ts = time.time()
    last_fps_push = 0.0
    frame_count = 0
    while not stop_event.is_set():
        try:
            while True:
                msg = settings_queue.get_nowait()
                if isinstance(msg, dict):
                    settings.update(msg)
                    if "core_vision" in msg:
                        _set_current_process_affinity(msg.get("core_vision"))
        except Exception:
            pass
        cfg = (
            max(160, int(settings.get("camera_raw_width", 640))),
            max(120, int(settings.get("camera_raw_height", 640))),
            max(5, int(settings.get("camera_fps", 60))),
            int(settings.get("sensor_bit_depth", 10)),
        )
        if current_cfg != cfg:
            try:
                if picam2 is not None:
                    picam2.stop()
                    picam2.close()
            except Exception:
                pass
            picam2 = None
            try:
                w, h, fps, bit_depth = cfg
                picam2 = Picamera2()
                picam2.start_preview(Preview.NULL)
                frame_duration = int(1000000 / fps)
                config = picam2.create_video_configuration(
                    sensor={"output_size": (w, h), "bit_depth": int(bit_depth)},
                    controls={"FrameDurationLimits": (frame_duration, frame_duration)}
                )
                config["main"]["format"] = "BGR888"
                config["main"]["size"] = (w, h)
                picam2.align_configuration(config)
                picam2.configure(config)
                picam2.start()
                current_cfg = cfg
            except Exception as exc:
                try:
                    status_queue.put_nowait(("vision_err", str(exc)))
                except Exception:
                    pass
                time.sleep(0.2)
                continue
        try:
            frame = picam2.capture_array()
        except Exception:
            time.sleep(0.01)
            continue
        try:
            x0, y0, cw, ch = _center_crop_rect(frame.shape[1], frame.shape[0], settings.get("video_crop_ratio", 1.0))
            frame = frame[y0 : y0 + ch, x0 : x0 + cw]
        except Exception:
            pass
        det = None
        used_roi = False
        try:
            roi_enabled = bool(settings.get("mp_roi_enabled", True))
        except Exception:
            roi_enabled = True
        if roi_enabled:
            try:
                fw = int(frame.shape[1])
                fh = int(frame.shape[0])
                roi_w = max(32, min(fw, int(settings.get("hough_crop_width", fw))))
                roi_h = max(32, min(fh, int(settings.get("hough_crop_height", fh))))
                valid = float(latest_det[5]) > 0.5
                stale_sec = float(settings.get("mp_roi_stale_sec", 0.25))
                age = time.time() - float(latest_det[4])
                if valid and age <= stale_sec and (roi_w < fw or roi_h < fh):
                    cx = float(latest_det[0])
                    cy = float(latest_det[1])
                    rx0 = max(0, min(fw - roi_w, int(round(cx - roi_w * 0.5))))
                    ry0 = max(0, min(fh - roi_h, int(round(cy - roi_h * 0.5))))
                    roi = frame[ry0 : ry0 + roi_h, rx0 : rx0 + roi_w]
                    det = _detect_circle_np(
                        roi,
                        ksize=settings.get("ksize", 5),
                        min_dist=settings.get("min_dist", 80),
                        param1=settings.get("param1", 220),
                        param2=settings.get("param2", 35),
                        min_radius=settings.get("min_radius", 20),
                        max_radius=settings.get("max_radius", 120),
                    )
                    used_roi = True
                    if det is not None:
                        det = (int(det[0]) + rx0, int(det[1]) + ry0, int(det[2]))
            except Exception:
                det = None
                used_roi = False
        if det is None:
            det = _detect_circle_np(
                frame,
                ksize=settings.get("ksize", 5),
                min_dist=settings.get("min_dist", 80),
                param1=settings.get("param1", 220),
                param2=settings.get("param2", 35),
                min_radius=settings.get("min_radius", 20),
                max_radius=settings.get("max_radius", 120),
            )
        ts = time.time()
        if det is not None:
            latest_det[0] = float(det[0])
            latest_det[1] = float(det[1])
            latest_det[2] = float(det[2])
            latest_det[3] = 1.0
            latest_det[4] = ts
            latest_det[5] = 1.0
        else:
            latest_det[5] = 0.0
            latest_det[4] = ts
        frame_count += 1
        push_hz = float(settings.get("preview_push_hz", 15.0))
        push_period = 1.0 / max(1.0, push_hz)
        if ts - last_fps_push >= push_period:
            hz = frame_count / max(1e-3, (ts - last_ts))
            frame_count = 0
            last_ts = ts
            last_fps_push = ts
            try:
                status_queue.put_nowait(("vision_hz", hz))
                if bool(settings.get("multiprocess_preview", True)):
                    show_debug = bool(settings.get("show_debug_panels", False))
                    if show_debug:
                        base = frame.copy()
                        k = int(settings.get("ksize", 5))
                        if k < 3:
                            k = 3
                        if k % 2 == 0:
                            k += 1
                        scale = 0.5
                        h0, w0 = base.shape[:2]
                        sw = max(2, int(round(w0 * scale)))
                        sh = max(2, int(round(h0 * scale)))
                        small = cv2.resize(base, (sw, sh), interpolation=cv2.INTER_AREA)
                        side = max(2, min(sw, sh))
                        x0 = max(0, (sw - side) // 2)
                        y0 = max(0, (sh - side) // 2)
                        square = small[y0 : y0 + side, x0 : x0 + side]
                        main_panel = square.copy()
                        green = square[:, :, 1]
                        red = square[:, :, 2] if square.shape[2] >= 3 else square[:, :, 0]
                        blurred_green = cv2.GaussianBlur(green, (k, k), 1)
                        blurred_red = cv2.GaussianBlur(red, (k, k), 1)

                        roi_x0 = 0
                        roi_y0 = 0
                        roi_x1 = side
                        roi_y1 = side
                        has_roi = False
                        if det is not None:
                            try:
                                fw = int(frame.shape[1])
                                fh = int(frame.shape[0])
                                roi_w = max(32, min(fw, int(settings.get("hough_crop_width", fw))))
                                roi_h = max(32, min(fh, int(settings.get("hough_crop_height", fh))))
                                cx = float(det[0])
                                cy = float(det[1])
                                rx0 = max(0, min(fw - roi_w, int(round(cx - roi_w * 0.5))))
                                ry0 = max(0, min(fh - roi_h, int(round(cy - roi_h * 0.5))))
                                rx1 = rx0 + roi_w
                                ry1 = ry0 + roi_h
                                roi_x0 = max(0, min(side, int(round(rx0 * scale)) - x0))
                                roi_y0 = max(0, min(side, int(round(ry0 * scale)) - y0))
                                roi_x1 = max(0, min(side, int(round(rx1 * scale)) - x0))
                                roi_y1 = max(0, min(side, int(round(ry1 * scale)) - y0))
                                if roi_x1 > roi_x0 + 1 and roi_y1 > roi_y0 + 1:
                                    has_roi = True
                            except Exception:
                                has_roi = False
                        if has_roi:
                            cv2.rectangle(main_panel, (roi_x0, roi_y0), (roi_x1 - 1, roi_y1 - 1), (0, 255, 255), 2)

                        green_panel = np.zeros((side, side, 3), dtype=np.uint8)
                        red_panel = np.zeros((side, side, 3), dtype=np.uint8)
                        bin_panel = np.zeros((side, side, 3), dtype=np.uint8)

                        if has_roi:
                            green_panel[roi_y0:roi_y1, roi_x0:roi_x1, 1] = blurred_green[roi_y0:roi_y1, roi_x0:roi_x1]
                            red_panel[roi_y0:roi_y1, roi_x0:roi_x1, 2] = blurred_red[roi_y0:roi_y1, roi_x0:roi_x1]
                            if bool(settings.get("laser_align_mode", False)):
                                laser_threshold = int(settings.get("laser_threshold", 180))
                                _, binary = cv2.threshold(blurred_red, laser_threshold, 255, cv2.THRESH_BINARY)
                                bin_panel[roi_y0:roi_y1, roi_x0:roi_x1] = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)[roi_y0:roi_y1, roi_x0:roi_x1]
                        else:
                            green_panel[:, :, 1] = blurred_green
                            red_panel[:, :, 2] = blurred_red
                            if bool(settings.get("laser_align_mode", False)):
                                laser_threshold = int(settings.get("laser_threshold", 180))
                                _, binary = cv2.threshold(blurred_red, laser_threshold, 255, cv2.THRESH_BINARY)
                                bin_panel = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
                        if det is not None:
                            dx = int(round(det[0] * scale)) - x0
                            dy = int(round(det[1] * scale)) - y0
                            dr = int(round(det[2] * scale))
                            if 0 <= dx < side and 0 <= dy < side:
                                cv2.circle(main_panel, (dx, dy), max(1, dr), (0, 0, 255), 2)
                                cv2.circle(main_panel, (dx, dy), 2, (0, 255, 0), -1)
                                cv2.circle(green_panel, (dx, dy), max(1, dr), (255, 0, 0), 2)
                                cv2.circle(red_panel, (dx, dy), 2, (255, 255, 255), -1)
                        cv2.putText(main_panel, "Main", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(green_panel, "Green", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(red_panel, "Red", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(bin_panel, "Bin", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
                        top = np.hstack((main_panel, green_panel))
                        bottom = np.hstack((red_panel, bin_panel))
                        preview = np.vstack((top, bottom))
                    else:
                        preview = frame.copy()
                        if det is not None:
                            cv2.circle(preview, (int(det[0]), int(det[1])), int(det[2]), (0, 0, 255), 2)
                            cv2.circle(preview, (int(det[0]), int(det[1])), 3, (0, 255, 0), -1)
                        ph = max(120, int(preview.shape[0] * 0.5))
                        pw = max(160, int(preview.shape[1] * 0.5))
                        preview = cv2.resize(preview, (pw, ph), interpolation=cv2.INTER_AREA)
                    ok, enc = cv2.imencode(".jpg", preview, [cv2.IMWRITE_JPEG_QUALITY, 70])
                    if ok:
                        status_queue.put_nowait(("preview_jpg", enc.tobytes()))
            except Exception:
                pass
    try:
        if picam2 is not None:
            picam2.stop()
            picam2.close()
    except Exception:
        pass


def _control_process_main(stop_event, settings_queue, cmd_queue, status_queue, latest_det):
    _set_current_process_affinity("3")
    settings = {
        "core_control": "3",
        "control_period_ms": 20,
        "control_debug_timing": False,
        "camera_raw_width": 640,
        "camera_raw_height": 640,
        "sensor_bit_depth": 10,
        "video_crop_ratio": 1.0,
        "track_enabled": True,
        "tracking_active": False,
        "pan_enabled": True,
        "tilt_enabled": True,
        "deadband": 3.0,
        "kp_x": 0.0075,
        "ki_x": 0.025,
        "kd_x": 0.000005,
        "kp_y": 0.01,
        "ki_y": 0.02,
        "kd_y": 0.000005,
        "pan_min": -360.0,
        "pan_max": 360.0,
        "tilt_min": -360.0,
        "tilt_max": 360.0,
        "brushless_pan_speed_dps": 120.0,
        "brushless_tilt_speed_dps": 120.0,
        "brushless_parallel_write": False,
        "imu_port": "/dev/ttyUSB0",
        "imu_baudrate": 9600,
        "imu_use_6axis": True,
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
        "auto_stabilize": False,
        "stab_gain_pitch": 1.0,
        "stab_gain_yaw": 1.0,
        "stab_pitch_deadband_deg": 0.6,
        "stab_yaw_deadband_deg": 0.6,
        "stab_tilt_limit_deg": 8.0,
        "stab_pan_limit_deg": 8.0,
        "stab_tilt_alpha": 0.35,
        "stab_pan_alpha": 0.35,
        "stab_tilt_rate_limit_deg_per_s": 120.0,
        "stab_pan_rate_limit_deg_per_s": 120.0,
        "stab_invert_x": False,
        "stab_invert_y": False,
        "stab_enable_pitch": True,
        "stab_enable_yaw": True,
    }
    pid_x = PID(kP=0.0075, kI=0.025, kD=0.000005, output_bound_low=-12, output_bound_high=12)
    pid_y = PID(kP=0.01, kI=0.02, kD=0.000005, output_bound_low=-12, output_bound_high=12)
    current_pan = 0.0
    current_tilt = 0.0
    servo = None
    servo_key = None
    imu = None
    imu_key = None
    imu_zero_pitch = 0.0
    imu_zero_yaw = 0.0
    stab_pan_filtered = 0.0
    stab_tilt_filtered = 0.0
    next_servo_retry_ts = 0.0
    servo_error_streak = 0
    last_ts = time.time()
    last_push = 0.0
    last_imu_push = 0.0
    last_auto_stabilize = False
    last_timing_push = 0.0
    jog_hold_until = 0.0
    while not stop_event.is_set():
        loop_start = time.time()
        try:
            while True:
                cmd = cmd_queue.get_nowait()
                if isinstance(cmd, tuple) and len(cmd) >= 1:
                    if cmd[0] == "recenter":
                        current_pan = 0.0
                        current_tilt = 0.0
                        pid_x.reset()
                        pid_y.reset()
                        stab_pan_filtered = 0.0
                        stab_tilt_filtered = 0.0
                    elif cmd[0] == "zero_imu":
                        imu_zero_pitch = float(cmd[1]) if len(cmd) > 1 else 0.0
                        imu_zero_yaw = float(cmd[2]) if len(cmd) > 2 else 0.0
                    elif cmd[0] == "zero_imu_request":
                        if imu is not None:
                            deadline = time.time() + 1.2
                            pitches = []
                            yaw_sin = 0.0
                            yaw_cos = 0.0
                            while time.time() < deadline and len(pitches) < 40 and not stop_event.is_set():
                                try:
                                    st = imu.get_state()
                                    if st.last_update > 0:
                                        pitches.append(float(st.pitch_deg))
                                        yr = math.radians(float(st.yaw_deg))
                                        yaw_sin += math.sin(yr)
                                        yaw_cos += math.cos(yr)
                                except Exception:
                                    pass
                                time.sleep(0.02)
                            if len(pitches) >= 5:
                                imu_zero_pitch = float(sum(pitches) / len(pitches))
                                imu_zero_yaw = float(math.degrees(math.atan2(yaw_sin, yaw_cos)))
                                try:
                                    status_queue.put_nowait(("imu_zero_base", (imu_zero_pitch, imu_zero_yaw)))
                                except Exception:
                                    pass
                    elif cmd[0] == "jog":
                        if len(cmd) >= 3:
                            current_pan += float(cmd[1])
                            current_tilt += float(cmd[2])
                            pid_x.reset()
                            pid_y.reset()
                            jog_hold_until = time.time() + 0.25
                    elif cmd[0] == "shutdown_pose":
                        current_pan = float(settings.get("shutdown_pan_deg", 0.0))
                        current_tilt = float(settings.get("shutdown_tilt_deg", 0.0))
        except Exception:
            pass
        t_cmd_end = time.time()
        try:
            while True:
                msg = settings_queue.get_nowait()
                if isinstance(msg, dict):
                    settings.update(msg)
                    if "core_control" in msg:
                        _set_current_process_affinity(msg.get("core_control"))
        except Exception:
            pass
        t_settings_end = time.time()
        loop_now = time.time()
        desired_key = (
            settings.get("brushless_pan_dev"),
            settings.get("brushless_tilt_dev"),
            settings.get("brushless_pan_baudrate"),
            settings.get("brushless_tilt_baudrate"),
            settings.get("brushless_pan_txden"),
            settings.get("brushless_tilt_txden"),
            settings.get("brushless_pan_direction_sign"),
            settings.get("brushless_tilt_direction_sign"),
            settings.get("pan_id"),
            settings.get("tilt_id"),
        )
        if (servo is None or servo_key != desired_key) and loop_now >= next_servo_retry_ts:
            try:
                if servo is not None:
                    servo.cleanup()
            except Exception:
                pass
            servo = None
            try:
                try:
                    import RPi.GPIO as _GPIO
                    _GPIO.setwarnings(False)
                    _GPIO.setmode(_GPIO.BCM)
                    _GPIO.cleanup(int(settings.get("brushless_pan_txden", 22)))
                    _GPIO.cleanup(int(settings.get("brushless_tilt_txden", 27)))
                except Exception:
                    pass
                base_dir = os.path.dirname(os.path.abspath(__file__))
                module = _load_module(
                    "_brushless_dual_driver_v1_mp",
                    os.path.join(base_dir, "brushless_motor", "dual_rs485_motor_driver_v1.py"),
                )
                servo = BrushlessDualServoAdapter(
                    motor_config_cls=module.MotorConfig,
                    motor_cls=module.LkMotor,
                    pan_id=int(settings.get("pan_id", 1)),
                    tilt_id=int(settings.get("tilt_id", 2)),
                    pan_dev=str(settings.get("brushless_pan_dev", "/dev/ttySC0")),
                    tilt_dev=str(settings.get("brushless_tilt_dev", "/dev/ttySC1")),
                    pan_baudrate=int(settings.get("brushless_pan_baudrate", 1000000)),
                    tilt_baudrate=int(settings.get("brushless_tilt_baudrate", 1000000)),
                    pan_txden=int(settings.get("brushless_pan_txden", 22)),
                    tilt_txden=int(settings.get("brushless_tilt_txden", 27)),
                    pan_direction_sign=int(settings.get("brushless_pan_direction_sign", 1)),
                    tilt_direction_sign=int(settings.get("brushless_tilt_direction_sign", 1)),
                    pan_speed_dps=float(settings.get("brushless_pan_speed_dps", 120.0)),
                    tilt_speed_dps=float(settings.get("brushless_tilt_speed_dps", 120.0)),
                    pan_min_deg=float(settings.get("pan_min", -360.0)),
                    pan_max_deg=float(settings.get("pan_max", 360.0)),
                    tilt_min_deg=float(settings.get("tilt_min", -360.0)),
                    tilt_max_deg=float(settings.get("tilt_max", 360.0)),
                )
                servo.set_angles(
                    [
                        (int(settings.get("pan_id", 1)), current_pan),
                        (int(settings.get("tilt_id", 2)), current_tilt),
                    ]
                )
                servo.move_angle(wait=False)
                servo_key = desired_key
                try:
                    status_queue.put_nowait(("control_info", "servo_connected"))
                except Exception:
                    pass
            except Exception as exc:
                servo = None
                servo_key = None
                next_servo_retry_ts = time.time() + 0.5
                try:
                    status_queue.put_nowait(("control_err", f"servo init: {exc}"))
                except Exception:
                    pass
        if servo is not None:
            try:
                servo.pan_speed_dps = float(settings.get("brushless_pan_speed_dps", 120.0))
                servo.tilt_speed_dps = float(settings.get("brushless_tilt_speed_dps", 120.0))
                if hasattr(servo, "parallel_write"):
                    servo.parallel_write = bool(settings.get("brushless_parallel_write", False))
            except Exception:
                pass
        t_servo_end = time.time()
        imu_desired_key = (
            settings.get("imu_port"),
            settings.get("imu_baudrate"),
            bool(settings.get("imu_use_6axis", True)),
            settings.get("imu_output_hz"),
            float(settings.get("imu_ax_offset_g", 0.0)),
            float(settings.get("imu_ay_offset_g", 0.0)),
            float(settings.get("imu_az_offset_g", 0.0)),
            float(settings.get("imu_gx_offset_dps", 0.0)),
            float(settings.get("imu_gy_offset_dps", 0.0)),
            float(settings.get("imu_gz_offset_dps", 0.0)),
            int(settings.get("imu_hx_offset", 0)),
            int(settings.get("imu_hy_offset", 0)),
            int(settings.get("imu_hz_offset", 0)),
            float(settings.get("imu_az_reference_g", 1.0)),
        )
        if imu is None or imu_key != imu_desired_key:
            try:
                if imu is not None:
                    imu.close()
            except Exception:
                pass
            imu = None
            try:
                imu = IMUReader(
                    port=str(settings.get("imu_port", "/dev/ttyUSB0")),
                    baudrate=int(settings.get("imu_baudrate", 9600)),
                    timeout=0.1,
                    debug=False,
                )
                try:
                    imu.configure_output(output_mask=0x001E, rate_code=0x08)
                except Exception:
                    pass
                try:
                    imu.set_algorithm_mode(bool(settings.get("imu_use_6axis", True)))
                except Exception:
                    pass
                try:
                    imu.set_output_rate_hz(int(settings.get("imu_output_hz", 50)))
                except Exception:
                    pass
                try:
                    imu.set_sensor_offsets(
                        ax_g=float(settings.get("imu_ax_offset_g", 0.0)),
                        ay_g=float(settings.get("imu_ay_offset_g", 0.0)),
                        az_g=float(settings.get("imu_az_offset_g", 0.0)),
                        gx_dps=float(settings.get("imu_gx_offset_dps", 0.0)),
                        gy_dps=float(settings.get("imu_gy_offset_dps", 0.0)),
                        gz_dps=float(settings.get("imu_gz_offset_dps", 0.0)),
                        hx=int(settings.get("imu_hx_offset", 0)),
                        hy=int(settings.get("imu_hy_offset", 0)),
                        hz=int(settings.get("imu_hz_offset", 0)),
                    )
                except Exception:
                    pass
                imu.start()
                imu_key = imu_desired_key
            except Exception as exc:
                imu = None
                imu_key = None
                try:
                    status_queue.put_nowait(("control_err", f"IMU init: {exc}"))
                except Exception:
                    pass
        t_imu_init_end = time.time()
        now = time.time()
        dt = max(1e-3, (now - last_ts))
        hz = 1.0 / dt
        last_ts = now
        valid = latest_det[5] > 0.5
        det_update_age = now - float(latest_det[4])
        age = det_update_age if valid else -1.0
        tracking_enabled = bool(settings.get("track_enabled", True)) and bool(settings.get("tracking_active", False))
        if now < jog_hold_until:
            tracking_enabled = False
        pid_x.set_gains(float(settings.get("kp_x", 0.0075)), float(settings.get("ki_x", 0.025)), float(settings.get("kd_x", 0.000005)))
        pid_y.set_gains(float(settings.get("kp_y", 0.01)), float(settings.get("ki_y", 0.02)), float(settings.get("kd_y", 0.000005)))
        delta_x = 0.0
        delta_y = 0.0
        if tracking_enabled and valid and age <= 0.3 and servo is not None:
            cx = float(latest_det[0])
            cy = float(latest_det[1])
            full_w = int(settings.get("camera_raw_width", 640))
            full_h = int(settings.get("camera_raw_height", 640))
            _, _, eff_w, eff_h = _center_crop_rect(full_w, full_h, float(settings.get("video_crop_ratio", 1.0)))
            target_x = float(eff_w) * 0.5
            target_y = float(eff_h) * 0.5
            error_x = target_x - cx
            error_y = target_y - cy
            deadband = float(settings.get("deadband", 3.0))
            if abs(error_x) <= deadband:
                error_x = 0.0
            if abs(error_y) <= deadband:
                error_y = 0.0
            delta_x = float(pid_x.update(error_x, dt=dt))
            delta_y = float(pid_y.update(error_y, dt=dt))
        else:
            pid_x.reset()
            pid_y.reset()
        stab_pan = 0.0
        stab_tilt = 0.0
        imu_pitch = 0.0
        imu_yaw = 0.0
        imu_age = -1.0
        auto_stabilize = bool(settings.get("auto_stabilize", False))
        had_imu_sample = False
        if imu is not None:
            try:
                imu_state = imu.get_state()
            except Exception as exc:
                imu_state = None
                try:
                    status_queue.put_nowait(("control_err", f"IMU read: {exc}"))
                except Exception:
                    pass
            if imu_state is not None and imu_state.last_update > 0:
                had_imu_sample = True
                imu_pitch = float(imu_state.pitch_deg)
                imu_yaw = float(imu_state.yaw_deg)
                imu_age = max(0.0, now - float(imu_state.last_update))
        if now - last_imu_push >= 0.10:
            try:
                status_queue.put_nowait(("imu_pitch", imu_pitch))
                status_queue.put_nowait(("imu_yaw", imu_yaw))
                status_queue.put_nowait(("imu_age", imu_age))
            except Exception:
                pass
            last_imu_push = now
        t_imu_read_end = time.time()
        if auto_stabilize and had_imu_sample:
                if not last_auto_stabilize:
                    imu_zero_pitch = imu_pitch
                    imu_zero_yaw = imu_yaw
                    stab_pan_filtered = 0.0
                    stab_tilt_filtered = 0.0
                    try:
                        status_queue.put_nowait(("imu_zero_base", (imu_zero_pitch, imu_zero_yaw)))
                    except Exception:
                        pass
                pitch_err = ((imu_pitch - imu_zero_pitch + 180.0) % 360.0) - 180.0
                yaw_err = ((imu_yaw - imu_zero_yaw + 180.0) % 360.0) - 180.0
                enable_pitch = bool(settings.get("stab_enable_pitch", True))
                enable_yaw = bool(settings.get("stab_enable_yaw", False))
                if bool(settings.get("stab_invert_y", False)):
                    pitch_err = -pitch_err
                if bool(settings.get("stab_invert_x", False)):
                    yaw_err = -yaw_err
                if not enable_pitch:
                    pitch_err = 0.0
                if not enable_yaw:
                    yaw_err = 0.0
                pitch_deadband = max(0.0, float(settings.get("stab_pitch_deadband_deg", 0.6)))
                yaw_deadband = max(0.0, float(settings.get("stab_yaw_deadband_deg", 0.6)))
                if abs(pitch_err) <= pitch_deadband:
                    pitch_err = 0.0
                else:
                    pitch_err = math.copysign(abs(pitch_err) - pitch_deadband, pitch_err)
                if abs(yaw_err) <= yaw_deadband:
                    yaw_err = 0.0
                else:
                    yaw_err = math.copysign(abs(yaw_err) - yaw_deadband, yaw_err)
                tilt_limit = max(0.0, float(settings.get("stab_tilt_limit_deg", 8.0)))
                pan_limit = max(0.0, float(settings.get("stab_pan_limit_deg", 8.0)))
                stab_tilt_target = max(-tilt_limit, min(tilt_limit, pitch_err * float(settings.get("stab_gain_pitch", 1.0))))
                stab_pan_target = max(-pan_limit, min(pan_limit, -yaw_err * float(settings.get("stab_gain_yaw", 1.0))))
                tilt_alpha = max(0.0, min(1.0, float(settings.get("stab_tilt_alpha", 0.35))))
                pan_alpha = max(0.0, min(1.0, float(settings.get("stab_pan_alpha", 0.35))))
                tilt_filtered_target = stab_tilt_filtered + tilt_alpha * (stab_tilt_target - stab_tilt_filtered)
                pan_filtered_target = stab_pan_filtered + pan_alpha * (stab_pan_target - stab_pan_filtered)
                tilt_rate = max(0.0, float(settings.get("stab_tilt_rate_limit_deg_per_s", 120.0)))
                pan_rate = max(0.0, float(settings.get("stab_pan_rate_limit_deg_per_s", 120.0)))
                tilt_delta_filtered = tilt_filtered_target - stab_tilt_filtered
                pan_delta_filtered = pan_filtered_target - stab_pan_filtered
                tilt_max_delta = tilt_rate * dt
                pan_max_delta = pan_rate * dt
                if tilt_delta_filtered > tilt_max_delta:
                    tilt_delta_filtered = tilt_max_delta
                elif tilt_delta_filtered < -tilt_max_delta:
                    tilt_delta_filtered = -tilt_max_delta
                if pan_delta_filtered > pan_max_delta:
                    pan_delta_filtered = pan_max_delta
                elif pan_delta_filtered < -pan_max_delta:
                    pan_delta_filtered = -pan_max_delta
                stab_tilt_filtered = stab_tilt_filtered + tilt_delta_filtered
                stab_pan_filtered = stab_pan_filtered + pan_delta_filtered
                stab_tilt = float(stab_tilt_filtered)
                stab_pan = float(stab_pan_filtered)
        last_auto_stabilize = bool(auto_stabilize and had_imu_sample)
        if servo is not None:
            pan_min = float(settings.get("pan_min", -360.0))
            pan_max = float(settings.get("pan_max", 360.0))
            tilt_min = float(settings.get("tilt_min", -360.0))
            tilt_max = float(settings.get("tilt_max", 360.0))
            if bool(settings.get("pan_enabled", True)):
                current_pan = max(pan_min, min(pan_max, current_pan + delta_x))
            if bool(settings.get("tilt_enabled", True)):
                current_tilt = max(tilt_min, min(tilt_max, current_tilt + delta_y))
            out_pan = float(current_pan + stab_pan)
            out_tilt = float(current_tilt + stab_tilt)
            out_pan = max(pan_min, min(pan_max, out_pan))
            out_tilt = max(tilt_min, min(tilt_max, out_tilt))
            try:
                servo.set_angles(
                    [
                        (int(settings.get("pan_id", 1)), out_pan),
                        (int(settings.get("tilt_id", 2)), out_tilt),
                    ]
                )
                servo.move_angle(wait=False)
                servo_error_streak = 0
            except Exception as exc:
                servo_error_streak += 1
                try:
                    status_queue.put_nowait(("control_err", f"servo write[{servo_error_streak}]: {exc}"))
                except Exception:
                    pass
                if servo_error_streak >= 3:
                    try:
                        servo.cleanup()
                    except Exception:
                        pass
                    servo = None
                    servo_key = None
                    next_servo_retry_ts = time.time() + 0.5
        t_servo_write_end = time.time()
        if now - last_push >= 0.25:
            try:
                status_queue.put_nowait(("control_hz", hz))
                status_queue.put_nowait(("det_age", age))
                status_queue.put_nowait(("det_valid", 1.0 if valid else 0.0))
                status_queue.put_nowait(("det_update_age", float(det_update_age)))
                if servo is not None:
                    status_queue.put_nowait(("control_pan", float(out_pan)))
                    status_queue.put_nowait(("control_tilt", float(out_tilt)))
                else:
                    status_queue.put_nowait(("control_pan", current_pan))
                    status_queue.put_nowait(("control_tilt", current_tilt))
            except Exception:
                pass
            last_push = now
        period_ms = max(1, int(settings.get("control_period_ms", 20)))
        work_ms = (t_servo_write_end - loop_start) * 1000.0
        sleep_ms = max(0.0, float(period_ms) - work_ms)
        debug_timing = bool(settings.get("control_debug_timing", False))
        if debug_timing and (now - last_timing_push) >= 0.8:
            try:
                status_queue.put_nowait(
                    (
                        "ctrl_timing",
                        {
                            "period_ms": float(period_ms),
                            "cmd_ms": (t_cmd_end - loop_start) * 1000.0,
                            "settings_ms": (t_settings_end - t_cmd_end) * 1000.0,
                            "servo_ms": (t_servo_end - t_settings_end) * 1000.0,
                            "imu_init_ms": (t_imu_init_end - t_servo_end) * 1000.0,
                            "imu_read_ms": (t_imu_read_end - t_imu_init_end) * 1000.0,
                            "servo_write_ms": (t_servo_write_end - t_imu_read_end) * 1000.0,
                            "work_ms": work_ms,
                            "sleep_ms": sleep_ms,
                        },
                    )
                )
            except Exception:
                pass
            last_timing_push = now
        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)
    try:
        if servo is not None:
            servo.cleanup()
    except Exception:
        pass
    try:
        if imu is not None:
            imu.close()
    except Exception:
        pass


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
        self._motor_config_cls = motor_config_cls
        self._motor_cls = motor_cls
        self._pan_cfg = {
            "name": "yaw",
            "motor_id": int(pan_id),
            "dev": str(pan_dev),
            "baudrate": int(pan_baudrate),
            "txden_pin": int(pan_txden),
            "direction_sign": int(pan_direction_sign),
            "default_speed_dps": float(pan_speed_dps),
            "min_deg": float(pan_min_deg),
            "max_deg": float(pan_max_deg),
        }
        self._tilt_cfg = {
            "name": "pitch",
            "motor_id": int(tilt_id),
            "dev": str(tilt_dev),
            "baudrate": int(tilt_baudrate),
            "txden_pin": int(tilt_txden),
            "direction_sign": int(tilt_direction_sign),
            "default_speed_dps": float(tilt_speed_dps),
            "min_deg": float(tilt_min_deg),
            "max_deg": float(tilt_max_deg),
        }
        self.pan_motor = self._motor_cls(self._motor_config_cls(**self._pan_cfg))
        self.tilt_motor = self._motor_cls(self._motor_config_cls(**self._tilt_cfg))
        self._pending_pan_deg = 0.0
        self._pending_tilt_deg = 0.0
        self.parallel_write = False
        self.pan_motor.motor_run()
        time.sleep(0.03)
        self.tilt_motor.motor_run()

    def _reset_axis_motor(self, axis: str):
        if axis == "pan":
            old = self.pan_motor
            cfg = self._pan_cfg
        else:
            old = self.tilt_motor
            cfg = self._tilt_cfg
        try:
            old.close()
        except Exception:
            pass
        m = self._motor_cls(self._motor_config_cls(**cfg))
        m.motor_run()
        time.sleep(0.03)
        if axis == "pan":
            self.pan_motor = m
        else:
            self.tilt_motor = m

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
        if bool(getattr(self, "parallel_write", False)):
            pan_exc = None
            tilt_exc = None

            def _move_pan():
                nonlocal pan_exc
                try:
                    self.pan_motor.move_to_deg(self._pending_pan_deg, max_speed_dps=self.pan_speed_dps)
                except Exception as exc:
                    pan_exc = exc

            def _move_tilt():
                nonlocal tilt_exc
                try:
                    self.tilt_motor.move_to_deg(self._pending_tilt_deg, max_speed_dps=self.tilt_speed_dps)
                except Exception as exc:
                    tilt_exc = exc

            t1 = threading.Thread(target=_move_pan, daemon=True)
            t2 = threading.Thread(target=_move_tilt, daemon=True)
            t1.start()
            time.sleep(0.002)
            t2.start()
            t1.join()
            t2.join()
            if pan_exc is not None:
                self._reset_axis_motor("pan")
            if tilt_exc is not None:
                self._reset_axis_motor("tilt")
            if pan_exc is not None or tilt_exc is not None:
                raise RuntimeError(f"brushless move failed: pan={pan_exc} tilt={tilt_exc}")
        else:
            try:
                self.pan_motor.move_to_deg(self._pending_pan_deg, max_speed_dps=self.pan_speed_dps)
            except Exception as exc:
                self._reset_axis_motor("pan")
                raise RuntimeError(f"brushless pan move failed: {exc}")
            time.sleep(0.005)
            try:
                self.tilt_motor.move_to_deg(self._pending_tilt_deg, max_speed_dps=self.tilt_speed_dps)
            except Exception as exc:
                self._reset_axis_motor("tilt")
                raise RuntimeError(f"brushless tilt move failed: {exc}")
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
            self.pan_motor.motor_off()
        except Exception:
            pass
        try:
            self.tilt_motor.motor_off()
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
        self.stab_thread = None
        self.mp_ctx = mp.get_context("spawn")
        self.mp_stop_event = None
        self.mp_vision_settings_queue = None
        self.mp_control_settings_queue = None
        self.mp_control_cmd_queue = None
        self.mp_status_queue = None
        self.mp_latest_detection = None
        self.mp_vision_proc = None
        self.mp_control_proc = None
        self.mp_control_hz = 0.0
        self.mp_vision_hz = 0.0
        self.mp_det_age = -1.0
        self.mp_det_update_age = -1.0
        self.mp_det_valid = 0.0
        self.mp_ctrl_timing = ""
        self.mp_pan = 0.0
        self.mp_tilt = 0.0
        self.mp_preview_jpg = None
        self.mp_preview_seq = 0
        self._last_render_seq = -1
        self._last_render_ts = 0.0
        self.mp_last_error = ""
        self.stop_event = threading.Event()
        self.detect_stop_event = threading.Event()
        self.stab_stop_event = threading.Event()
        self.camera_reconfigure_event = threading.Event()
        self.frame_queue = queue.Queue(maxsize=1)
        self.settings_lock = threading.Lock()
        self.detect_lock = threading.Lock()
        self.motion_lock = threading.Lock()
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
        self.stab_pan_comp_deg = 0.0
        self.stab_tilt_comp_deg = 0.0
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
        self.brushless_parallel_write = tk.BooleanVar(value=False)
        self.shutdown_pan_deg = tk.DoubleVar(value=0.0)
        self.shutdown_tilt_deg = tk.DoubleVar(value=0.0)
        self.shutdown_speed_dps = tk.DoubleVar(value=60.0)
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
        self.control_period_ms = tk.IntVar(value=50)
        self.stab_period_ms = tk.IntVar(value=12)
        self.control_debug_timing = tk.BooleanVar(value=True)
        self.multiprocess_mode = tk.BooleanVar(value=True)
        self.core_affinity_enabled = tk.BooleanVar(value=False)
        self.core_ui = tk.IntVar(value=0)
        self.core_control = tk.IntVar(value=1)
        self.core_vision = tk.StringVar(value="2")
        self.core_stabilize = tk.IntVar(value=3)
        self.core_layout_preset = tk.StringVar(value="标准版(UI0 视1-2 控3)")
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
        self.exposure_value = tk.IntVar(value=10000) # 微秒 us
        self.analogue_gain = tk.DoubleVar(value=1.0)
        self.ae_enable = tk.BooleanVar(value=True)
        self.ksize = tk.IntVar(value=5)
        self.min_dist = tk.IntVar(value=80)
        self.param1 = tk.IntVar(value=220)
        self.param2 = tk.IntVar(value=35)
        self.min_radius = tk.IntVar(value=20)
        self.max_radius = tk.IntVar(value=120)
        self.detect_ema_enabled = tk.BooleanVar(value=True)
        self.detect_ema_alpha = tk.DoubleVar(value=0.3)
        self.x_bias = tk.IntVar(value=0)
        self.y_bias = tk.IntVar(value=0)
        self.camera_fps = tk.IntVar(value=120)
        self.ui_refresh_hz = tk.IntVar(value=30)
        self.preview_push_hz = tk.IntVar(value=15)
        self.preview_render_hz = tk.IntVar(value=20)
        self.camera_raw_width = tk.IntVar(value=640)
        self.camera_raw_height = tk.IntVar(value=640)
        self.sensor_bit_depth = tk.IntVar(value=10)
        self.video_crop_ratio = tk.DoubleVar(value=1.0)
        self.hough_crop_width = tk.IntVar(value=640)
        self.hough_crop_height = tk.IntVar(value=640)
        self.image_rotate_deg = tk.DoubleVar(value=0.0)
        self.status_text = tk.StringVar(value="就绪")
        self.status_log_widget = None
        self.show_debug_panels = tk.BooleanVar(value=False)
        self.aggressive_perf_mode = tk.BooleanVar(value=False)
        self.multiprocess_preview = tk.BooleanVar(value=True)

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
        self.stab_invert_x = tk.BooleanVar(value=False)
        self.stab_invert_y = tk.BooleanVar(value=False)
        self.stab_enable_pitch = tk.BooleanVar(value=True)
        self.stab_enable_yaw = tk.BooleanVar(value=True)
        
        # 激光指示对准配置
        self.laser_align_mode = tk.BooleanVar(value=False) # False:盲对准, True:指示对准
        self.laser_threshold = tk.IntVar(value=240)        # 激光二值化阈值
        
        # 舵机角度范围配置（角度制）
        self.pan_min = tk.DoubleVar(value=-360.0)
        self.pan_max = tk.DoubleVar(value=360.0)
        self.tilt_min = tk.DoubleVar(value=-360.0)
        self.tilt_max = tk.DoubleVar(value=360.0)
        
        # 卡尔曼滤波参数
        self.kalman_enabled = tk.BooleanVar(value=True)
        self.kalman_process_noise = tk.DoubleVar(value=0.03)
        self.kalman_measurement_noise = tk.DoubleVar(value=0.4)
        
        # 硬件物理边界（从舵机读取）
        self.hw_pan_min = tk.DoubleVar(value=-360.0)
        self.hw_pan_max = tk.DoubleVar(value=360.0)
        self.hw_tilt_min = tk.DoubleVar(value=-360.0)
        self.hw_tilt_max = tk.DoubleVar(value=360.0)
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

        self._autosave_after_id = None
        self._autosave_suppress = True
        self._is_closing = False

        self._load_settings()
        init_w, init_h, _ = self._normalize_camera_resolution(self.camera_raw_width.get(), self.camera_raw_height.get())
        self.camera_raw_width.set(init_w)
        self.camera_raw_height.set(init_h)
        self.camera_target_size = (init_w, init_h)
        self._build_ui()
        self._preview_box_size = None
        self._autosize_window()
        try:
            self.preview_container.bind("<Configure>", self._on_preview_container_configure)
        except Exception:
            pass
        self.status_text.trace_add("write", self._on_status_text_changed)
        self.servo_mode.trace_add("write", self._on_servo_mode_change)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.root.bind("<Escape>", lambda _e: self.on_close())
        self.root.bind("q", lambda _e: self.on_close())
        self._update_settings_from_vars()
        self._attach_autosave()
        self._autosave_suppress = False
        self.laser_ranger = None
        self.root.after(10, self._start_runtime)
        self._startup_hardware_prep_async()

    def _startup_hardware_prep_async(self):
        mp_mode = False
        try:
            mp_mode = bool(self.multiprocess_mode.get())
        except Exception:
            mp_mode = False
        def _worker():
            try:
                try:
                    self.root.after(0, lambda: self.status_text.set("初始化外设中..."))
                except Exception:
                    pass
                try:
                    configure_laser_module(
                        port="/dev/ttyAMA3",
                        baudrate=115200,
                        module_id=0,
                        output_mode="inquire",
                        range_mode="medium",
                        interface_mode="uart",
                        uart_baudrate=115200,
                    )
                except Exception as exc:
                    print(f"[WARNING] Laser configure failed: {exc}")
                time.sleep(1.0)
                if not mp_mode:
                    try:
                        fallback_mode = "无刷RS485"
                        try:
                            self.root.after(0, lambda: self.servo_mode.set(fallback_mode))
                        except Exception:
                            self.servo_mode.set(fallback_mode)
                        self._ensure_servo()
                        try:
                            self._ensure_imu()
                            self._zero_imu()
                        except Exception as exc:
                            print(f"[WARNING] IMU init/zero failed before start: {exc}")
                    except Exception as exc:
                        import traceback

                        traceback.print_exc()
                        print(f"[WARNING] Servo init failed before start: {exc}")
                time.sleep(1.0)
                try:
                    self.laser_ranger = LaserRangerQueryMonitor(
                        port="/dev/ttyAMA3", baudrate=115200, module_id=0, history_len=10
                    )
                except Exception as exc:
                    print(f"[WARNING] Laser Ranger init failed: {exc}")
            finally:
                try:
                    self._release_local_hardware_handles()
                except Exception:
                    pass
                try:
                    self.root.after(0, lambda: self.status_text.set(self.status_text.get()))
                except Exception:
                    pass

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

    def _autosize_window(self):
        try:
            sw = int(self.root.winfo_screenwidth())
            sh = int(self.root.winfo_screenheight())
        except Exception:
            return
        w = max(900, int(sw * 0.92))
        h = max(650, int(sh * 0.88))
        x = max(0, (sw - w) // 2)
        y = max(0, (sh - h) // 2)
        try:
            self.root.geometry(f"{w}x{h}+{x}+{y}")
        except Exception:
            pass
        try:
            self.root.minsize(900, 650)
        except Exception:
            pass

    def _on_preview_container_configure(self, event):
        try:
            w = int(event.width)
            h = int(event.height)
        except Exception:
            return
        side = max(64, min(w, h))
        try:
            self.preview_frame.place_configure(
                x=(w - side) // 2,
                y=(h - side) // 2,
                width=side,
                height=side,
            )
        except Exception:
            pass
        self._preview_box_size = (side, side)

    def _ui_refresh_delay_ms(self):
        try:
            hz = int(self.ui_refresh_hz.get())
        except Exception:
            hz = 30
        hz = max(1, hz)
        return max(5, int(round(1000.0 / float(hz))))

    def _make_scrollable_tab(self, parent):
        container = ttk.Frame(parent)
        container.pack(fill=tk.BOTH, expand=True)
        canvas = tk.Canvas(container, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        inner = ttk.Frame(canvas, padding=8)
        window_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_inner_configure(_event):
            try:
                canvas.configure(scrollregion=canvas.bbox("all"))
            except Exception:
                pass

        def _on_canvas_configure(event):
            try:
                canvas.itemconfigure(window_id, width=int(event.width))
            except Exception:
                pass

        def _on_mousewheel(event):
            try:
                delta = int(getattr(event, "delta", 0))
                if delta == 0:
                    return
                if abs(delta) < 120:
                    step = -1 if delta > 0 else 1
                else:
                    step = int(-delta / 120)
                canvas.yview_scroll(step, "units")
            except Exception:
                pass

        def _on_button4(_event):
            try:
                canvas.yview_scroll(-1, "units")
            except Exception:
                pass

        def _on_button5(_event):
            try:
                canvas.yview_scroll(1, "units")
            except Exception:
                pass

        inner.bind("<Configure>", _on_inner_configure)
        canvas.bind("<Configure>", _on_canvas_configure)
        canvas.bind("<MouseWheel>", _on_mousewheel)
        canvas.bind("<Button-4>", _on_button4)
        canvas.bind("<Button-5>", _on_button5)
        return inner

    def _release_local_hardware_handles(self):
        if self.servo is not None:
            try:
                self.servo.cleanup()
            except Exception:
                pass
            self.servo = None
        if self.imu is not None:
            try:
                self.imu.close()
            except Exception:
                pass
            self.imu = None
        try:
            import RPi.GPIO as _GPIO

            _GPIO.setwarnings(False)
            try:
                if _GPIO.getmode() is None:
                    _GPIO.setmode(_GPIO.BCM)
            except Exception:
                _GPIO.setmode(_GPIO.BCM)
            try:
                settings = self._get_settings()
                _GPIO.cleanup(int(settings.get("brushless_pan_txden", 22)))
                _GPIO.cleanup(int(settings.get("brushless_tilt_txden", 27)))
            except Exception:
                pass
        except Exception:
            pass

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
            "shutdown_pan_deg": 0.0,
            "shutdown_tilt_deg": 0.0,
            "shutdown_speed_dps": 60.0,
            "imu_port": "/dev/ttyUSB0",
            "imu_baudrate": 9600,
            "imu_use_6axis": True,
            "imu_output_hz": 50,
            "pan_id": 1,
            "tilt_id": 2,
            "control_period_ms": 50,
            "stab_period_ms": 12,
            "control_debug_timing": True,
            "multiprocess_mode": True,
            "core_affinity_enabled": False,
            "core_ui": 0,
            "core_control": 1,
            "core_vision": "2",
            "core_stabilize": 3,
            "kp_x": 0.0075,
            "ki_x": 0.025,
            "kd_x": 0.000005,
            "kp_y": 0.01,
            "ki_y": 0.02,
            "kd_y": 0.000005,
            "deadband": 3.0,
            "exposure": 0.0,
            "gain": 8.0,
            "ksize": 5,
            "min_dist": 80,
            "param1": 220,
            "param2": 35,
            "min_radius": 20,
            "max_radius": 120,
            "detect_ema_enabled": True,
            "detect_ema_alpha": 0.3,
            "x_bias": 0,
            "y_bias": 0,
            "camera_fps": 120,
            "ui_refresh_hz": 30,
            "preview_push_hz": 15,
            "camera_raw_width": 640,
            "camera_raw_height": 640,
            "sensor_bit_depth": 10,
            "video_crop_ratio": 1.0,
            "hough_crop_width": 640,
            "hough_crop_height": 640,
            "image_rotate_deg": 0.0,
            "show_debug_panels": False,
            "aggressive_perf_mode": False,
            "show_debug_panels": False,
            "aggressive_perf_mode": False,
            "laser_align_mode": False,
            "laser_threshold": 240,
            "pan_min": -360.0,
            "pan_max": 360.0,
            "tilt_min": -360.0,
            "tilt_max": 360.0,
            "hw_pan_min": -360.0,
            "hw_pan_max": 360.0,
            "hw_tilt_min": -360.0,
            "hw_tilt_max": 360.0,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.4,
            "kalman_enabled": True,
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
            "stab_invert_x": False,
            "stab_invert_y": False,
            "stab_enable_pitch": True,
            "stab_enable_yaw": True,
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
                "brushless_parallel_write": safe_bool(self.brushless_parallel_write),
                "shutdown_pan_deg": safe_float(self.shutdown_pan_deg, "shutdown_pan_deg"),
                "shutdown_tilt_deg": safe_float(self.shutdown_tilt_deg, "shutdown_tilt_deg"),
                "shutdown_speed_dps": safe_float(self.shutdown_speed_dps, "shutdown_speed_dps"),
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
                "control_period_ms": safe_int(self.control_period_ms, "control_period_ms"),
                "stab_period_ms": safe_int(self.stab_period_ms, "stab_period_ms"),
                "control_debug_timing": safe_bool(self.control_debug_timing),
                "multiprocess_mode": safe_bool(self.multiprocess_mode),
                "core_affinity_enabled": safe_bool(self.core_affinity_enabled),
                "core_ui": safe_int(self.core_ui, "core_ui"),
                "core_control": safe_int(self.core_control, "core_control"),
                "core_vision": str(self.core_vision.get()).strip() if str(self.core_vision.get()).strip() else str(defaults["core_vision"]),
                "core_stabilize": safe_int(self.core_stabilize, "core_stabilize"),
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
                "exposure": safe_float(self.exposure_value, "exposure"),
                "gain": safe_float(self.analogue_gain, "gain"),
                "ae_enable": safe_bool(self.ae_enable),
                "ksize": safe_int(self.ksize, "ksize"),
                "min_dist": safe_int(self.min_dist, "min_dist"),
                "param1": safe_int(self.param1, "param1"),
                "param2": safe_int(self.param2, "param2"),
                "min_radius": safe_int(self.min_radius, "min_radius"),
                "max_radius": safe_int(self.max_radius, "max_radius"),
                "detect_ema_enabled": safe_bool(self.detect_ema_enabled),
                "detect_ema_alpha": safe_float(self.detect_ema_alpha, "detect_ema_alpha"),
                "x_bias": safe_int(self.x_bias, "x_bias"),
                "y_bias": safe_int(self.y_bias, "y_bias"),
                "camera_fps": safe_int(self.camera_fps, "camera_fps"),
                "ui_refresh_hz": safe_int(self.ui_refresh_hz, "ui_refresh_hz"),
                "preview_push_hz": safe_int(self.preview_push_hz, "preview_push_hz"),
                "camera_raw_width": safe_int(self.camera_raw_width, "camera_raw_width"),
                "camera_raw_height": safe_int(self.camera_raw_height, "camera_raw_height"),
                "sensor_bit_depth": safe_int(self.sensor_bit_depth, "sensor_bit_depth"),
                "video_crop_ratio": safe_float(self.video_crop_ratio, "video_crop_ratio"),
                "hough_crop_width": safe_int(self.hough_crop_width, "hough_crop_width"),
                "hough_crop_height": safe_int(self.hough_crop_height, "hough_crop_height"),
                "image_rotate_deg": safe_float(self.image_rotate_deg, "image_rotate_deg"),
                "show_debug_panels": safe_bool(self.show_debug_panels),
                "aggressive_perf_mode": safe_bool(self.aggressive_perf_mode),
                "laser_align_mode": safe_bool(self.laser_align_mode),
                "laser_threshold": safe_int(self.laser_threshold, "laser_threshold"),
                "pan_min": safe_float(self.pan_min, "pan_min"),
                "pan_max": safe_float(self.pan_max, "pan_max"),
            "tilt_min": safe_float(self.tilt_min, "tilt_min"),
            "tilt_max": safe_float(self.tilt_max, "tilt_max"),
            "kalman_process_noise": safe_float(self.kalman_process_noise, "kalman_process_noise"),
            "kalman_measurement_noise": safe_float(self.kalman_measurement_noise, "kalman_measurement_noise"),
                "kalman_enabled": safe_bool(self.kalman_enabled),
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
                "stab_invert_x": safe_bool(self.stab_invert_x),
                "stab_invert_y": safe_bool(self.stab_invert_y),
                "stab_enable_pitch": safe_bool(self.stab_enable_pitch),
                "stab_enable_yaw": safe_bool(self.stab_enable_yaw),
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
            "shutdown_pan_deg": 0.0,
            "shutdown_tilt_deg": 0.0,
            "shutdown_speed_dps": 60.0,
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
            "control_period_ms": 50,
            "stab_period_ms": 12,
            "multiprocess_mode": True,
            "core_affinity_enabled": False,
            "core_ui": 0,
            "core_control": 1,
            "core_vision": "2",
            "core_stabilize": 3,
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
            "exposure": 10000,
            "gain": 1.0,
            "ae_enable": True,
            "ksize": 5,
            "min_dist": 80,
            "param1": 220,
            "param2": 35,
            "min_radius": 20,
            "max_radius": 120,
            "detect_ema_enabled": True,
            "detect_ema_alpha": 0.3,
            "x_bias": 0,
            "y_bias": 0,
            "camera_fps": 120,
            "ui_refresh_hz": 30,
            "preview_push_hz": 15,
            "camera_raw_width": 640,
            "camera_raw_height": 640,
            "sensor_bit_depth": 10,
            "video_crop_ratio": 1.0,
            "hough_crop_width": 640,
            "hough_crop_height": 640,
            "image_rotate_deg": 0.0,
            "show_debug_panels": False,
            "aggressive_perf_mode": False,
            "pan_min": -360.0,
            "pan_max": 360.0,
            "tilt_min": -360.0,
            "tilt_max": 360.0,
            "kalman_process_noise": 0.03,
            "kalman_measurement_noise": 0.4,
            "kalman_enabled": True,
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
            "stab_invert_x": False,
            "stab_invert_y": False,
            "stab_enable_pitch": True,
            "stab_enable_yaw": False,
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
            "brushless_parallel_write": safe_bool(self.brushless_parallel_write, "brushless_parallel_write"),
            "shutdown_pan_deg": safe_float(self.shutdown_pan_deg, "shutdown_pan_deg"),
            "shutdown_tilt_deg": safe_float(self.shutdown_tilt_deg, "shutdown_tilt_deg"),
            "shutdown_speed_dps": safe_float(self.shutdown_speed_dps, "shutdown_speed_dps"),
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
            "control_period_ms": safe_int(self.control_period_ms, "control_period_ms"),
            "stab_period_ms": safe_int(self.stab_period_ms, "stab_period_ms"),
            "control_debug_timing": safe_bool(self.control_debug_timing, "control_debug_timing"),
            "multiprocess_mode": safe_bool(self.multiprocess_mode, "multiprocess_mode"),
            "core_affinity_enabled": safe_bool(self.core_affinity_enabled, "core_affinity_enabled"),
            "core_ui": safe_int(self.core_ui, "core_ui"),
            "core_control": safe_int(self.core_control, "core_control"),
            "core_vision": str(self.core_vision.get()).strip() if str(self.core_vision.get()).strip() else str(defaults["core_vision"]),
            "core_stabilize": safe_int(self.core_stabilize, "core_stabilize"),
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
            "exposure": safe_int(self.exposure_value, "exposure"),
            "gain": safe_float(self.analogue_gain, "gain"),
            "ae_enable": safe_bool(self.ae_enable, "ae_enable"),
            "ksize": safe_int(self.ksize, "ksize"),
            "min_dist": safe_int(self.min_dist, "min_dist"),
            "param1": safe_int(self.param1, "param1"),
            "param2": safe_int(self.param2, "param2"),
            "min_radius": safe_int(self.min_radius, "min_radius"),
            "max_radius": safe_int(self.max_radius, "max_radius"),
            "detect_ema_enabled": safe_bool(self.detect_ema_enabled, "detect_ema_enabled"),
            "detect_ema_alpha": safe_float(self.detect_ema_alpha, "detect_ema_alpha"),
            "x_bias": safe_int(self.x_bias, "x_bias"),
            "y_bias": safe_int(self.y_bias, "y_bias"),
            "camera_fps": safe_int(self.camera_fps, "camera_fps"),
            "ui_refresh_hz": safe_int(self.ui_refresh_hz, "ui_refresh_hz"),
            "preview_push_hz": safe_int(self.preview_push_hz, "preview_push_hz"),
            "camera_raw_width": safe_int(self.camera_raw_width, "camera_raw_width"),
            "camera_raw_height": safe_int(self.camera_raw_height, "camera_raw_height"),
            "sensor_bit_depth": safe_int(self.sensor_bit_depth, "sensor_bit_depth"),
            "video_crop_ratio": safe_float(self.video_crop_ratio, "video_crop_ratio"),
            "hough_crop_width": safe_int(self.hough_crop_width, "hough_crop_width"),
            "hough_crop_height": safe_int(self.hough_crop_height, "hough_crop_height"),
            "image_rotate_deg": safe_float(self.image_rotate_deg, "image_rotate_deg"),
            "show_debug_panels": safe_bool(self.show_debug_panels, "show_debug_panels"),
            "aggressive_perf_mode": safe_bool(self.aggressive_perf_mode, "aggressive_perf_mode"),
            "laser_align_mode": safe_bool(self.laser_align_mode, "laser_align_mode"),
            "laser_threshold": safe_int(self.laser_threshold, "laser_threshold"),
            "pan_min": safe_float(self.pan_min, "pan_min"),
            "pan_max": safe_float(self.pan_max, "pan_max"),
            "tilt_min": safe_float(self.tilt_min, "tilt_min"),
            "tilt_max": safe_float(self.tilt_max, "tilt_max"),
            "kalman_process_noise": safe_float(self.kalman_process_noise, "kalman_process_noise"),
            "kalman_measurement_noise": safe_float(self.kalman_measurement_noise, "kalman_measurement_noise"),
            "kalman_enabled": safe_bool(self.kalman_enabled, "kalman_enabled"),
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
            "stab_invert_x": safe_bool(self.stab_invert_x, "stab_invert_x"),
            "stab_invert_y": safe_bool(self.stab_invert_y, "stab_invert_y"),
            "stab_enable_pitch": safe_bool(self.stab_enable_pitch, "stab_enable_pitch"),
            "stab_enable_yaw": safe_bool(self.stab_enable_yaw, "stab_enable_yaw"),
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
            "shutdown_pan_deg": self.shutdown_pan_deg,
            "shutdown_tilt_deg": self.shutdown_tilt_deg,
            "shutdown_speed_dps": self.shutdown_speed_dps,
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
            "control_period_ms": self.control_period_ms,
            "stab_period_ms": self.stab_period_ms,
            "control_debug_timing": self.control_debug_timing,
            "multiprocess_mode": self.multiprocess_mode,
            "core_affinity_enabled": self.core_affinity_enabled,
            "core_ui": self.core_ui,
            "core_control": self.core_control,
            "core_vision": self.core_vision,
            "core_stabilize": self.core_stabilize,
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
            "exposure": self.exposure_value,
            "gain": self.analogue_gain,
            "ae_enable": self.ae_enable,
            "ksize": self.ksize,
            "min_dist": self.min_dist,
            "param1": self.param1,
            "param2": self.param2,
            "min_radius": self.min_radius,
            "max_radius": self.max_radius,
            "detect_ema_enabled": self.detect_ema_enabled,
            "detect_ema_alpha": self.detect_ema_alpha,
            "x_bias": self.x_bias,
            "y_bias": self.y_bias,
            "camera_fps": self.camera_fps,
            "ui_refresh_hz": self.ui_refresh_hz,
            "preview_push_hz": self.preview_push_hz,
            "camera_raw_width": self.camera_raw_width,
            "camera_raw_height": self.camera_raw_height,
            "sensor_bit_depth": self.sensor_bit_depth,
            "video_crop_ratio": self.video_crop_ratio,
            "hough_crop_width": self.hough_crop_width,
            "hough_crop_height": self.hough_crop_height,
            "image_rotate_deg": self.image_rotate_deg,
            "show_debug_panels": self.show_debug_panels,
            "aggressive_perf_mode": self.aggressive_perf_mode,
            "pan_min": self.pan_min,
            "pan_max": self.pan_max,
            "tilt_min": self.tilt_min,
            "tilt_max": self.tilt_max,
            "kalman_process_noise": self.kalman_process_noise,
            "kalman_measurement_noise": self.kalman_measurement_noise,
            "kalman_enabled": self.kalman_enabled,
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
            "stab_invert_x": self.stab_invert_x,
            "stab_invert_y": self.stab_invert_y,
            "stab_enable_pitch": self.stab_enable_pitch,
            "stab_enable_yaw": self.stab_enable_yaw,
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
                "brushless_parallel_write": bool(self.brushless_parallel_write.get()),
                "shutdown_pan_deg": float(self.shutdown_pan_deg.get()),
                "shutdown_tilt_deg": float(self.shutdown_tilt_deg.get()),
                "shutdown_speed_dps": float(self.shutdown_speed_dps.get()),
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
                "control_period_ms": int(self.control_period_ms.get()),
                "stab_period_ms": int(self.stab_period_ms.get()),
                "control_debug_timing": bool(self.control_debug_timing.get()),
                "multiprocess_mode": bool(self.multiprocess_mode.get()),
                "core_affinity_enabled": bool(self.core_affinity_enabled.get()),
                "core_ui": int(self.core_ui.get()),
                "core_control": int(self.core_control.get()),
                "core_vision": str(self.core_vision.get()).strip(),
                "core_stabilize": int(self.core_stabilize.get()),
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
                "exposure": float(self.exposure_value.get()),
                "gain": float(self.analogue_gain.get()),
                "ae_enable": bool(self.ae_enable.get()),
                "ksize": int(self.ksize.get()),
                "min_dist": int(self.min_dist.get()),
                "param1": int(self.param1.get()),
                "param2": int(self.param2.get()),
                "min_radius": int(self.min_radius.get()),
                "max_radius": int(self.max_radius.get()),
                "detect_ema_enabled": bool(self.detect_ema_enabled.get()),
                "detect_ema_alpha": float(self.detect_ema_alpha.get()),
                "x_bias": int(self.x_bias.get()),
                "y_bias": int(self.y_bias.get()),
                "camera_fps": int(self.camera_fps.get()),
                "ui_refresh_hz": int(self.ui_refresh_hz.get()),
                "preview_push_hz": int(self.preview_push_hz.get()),
                "camera_raw_width": int(self.camera_raw_width.get()),
                "camera_raw_height": int(self.camera_raw_height.get()),
                "sensor_bit_depth": int(self.sensor_bit_depth.get()),
                "video_crop_ratio": float(self.video_crop_ratio.get()),
                "hough_crop_width": int(self.hough_crop_width.get()),
                "hough_crop_height": int(self.hough_crop_height.get()),
                "image_rotate_deg": float(self.image_rotate_deg.get()),
                "show_debug_panels": bool(self.show_debug_panels.get()),
                "aggressive_perf_mode": bool(self.aggressive_perf_mode.get()),
                "show_debug_panels": bool(self.show_debug_panels.get()),
                "aggressive_perf_mode": bool(self.aggressive_perf_mode.get()),
                "laser_align_mode": bool(self.laser_align_mode.get()),
                "laser_threshold": int(self.laser_threshold.get()),
                "pan_min": float(self.pan_min.get()),
                "pan_max": float(self.pan_max.get()),
                "tilt_min": float(self.tilt_min.get()),
                "tilt_max": float(self.tilt_max.get()),
                "kalman_process_noise": float(self.kalman_process_noise.get()),
                "kalman_measurement_noise": float(self.kalman_measurement_noise.get()),
                "kalman_enabled": bool(self.kalman_enabled.get()),
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
                "stab_invert_x": bool(self.stab_invert_x.get()),
                "stab_invert_y": bool(self.stab_invert_y.get()),
                "stab_enable_pitch": bool(self.stab_enable_pitch.get()),
                "stab_enable_yaw": bool(self.stab_enable_yaw.get()),
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
            self.shutdown_pan_deg,
            self.shutdown_tilt_deg,
            self.shutdown_speed_dps,
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
            self.control_period_ms,
            self.stab_period_ms,
            self.control_debug_timing,
            self.brushless_parallel_write,
            self.multiprocess_mode,
            self.core_affinity_enabled,
            self.core_ui,
            self.core_control,
            self.core_vision,
            self.core_stabilize,
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
            self.exposure_value,
            self.analogue_gain,
            self.ae_enable,
            self.ksize,
            self.min_dist,
            self.param1,
            self.param2,
            self.min_radius,
            self.max_radius,
            self.detect_ema_enabled,
            self.detect_ema_alpha,
            self.x_bias,
            self.y_bias,
            self.camera_raw_width,
            self.camera_raw_height,
            self.sensor_bit_depth,
            self.video_crop_ratio,
            self.hough_crop_width,
            self.hough_crop_height,
            self.image_rotate_deg,
            self.show_debug_panels,
            self.aggressive_perf_mode,
            self.camera_fps,
            self.ui_refresh_hz,
            self.preview_push_hz,
            self.laser_align_mode,
            self.laser_threshold,
            self.pan_min,
            self.pan_max,
            self.tilt_min,
            self.tilt_max,
            self.kalman_process_noise,
            self.kalman_measurement_noise,
            self.kalman_enabled,
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
            self.stab_invert_x,
            self.stab_invert_y,
            self.stab_enable_pitch,
            self.stab_enable_yaw,
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

        self.preview_container = ttk.Frame(right)
        self.preview_container.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        try:
            self.preview_container.pack_propagate(False)
        except Exception:
            pass
        self.preview_frame = ttk.Frame(self.preview_container)
        self.preview_frame.place(x=0, y=0, width=320, height=320)
        try:
            self.preview_frame.pack_propagate(False)
        except Exception:
            pass
        self.preview_label = ttk.Label(self.preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=True)
        ttk.Label(right, textvariable=self.status_text).pack(anchor=tk.W, pady=(6, 0))
        ttk.Checkbutton(
            right,
            text="更加激进(关闭图像显示, 提升处理速度)",
            variable=self.aggressive_perf_mode,
            command=self._on_aggressive_perf_mode_toggle,
        ).pack(anchor=tk.W, pady=(2, 0))
        ttk.Checkbutton(
            right,
            text="多进程显示预览",
            variable=self.multiprocess_preview,
        ).pack(anchor=tk.W, pady=(2, 0))
        self.status_log_widget = scrolledtext.ScrolledText(right, height=6, wrap=tk.WORD)
        self.status_log_widget.pack(fill=tk.X, pady=(4, 0))
        self.status_log_widget.configure(state=tk.DISABLED)

        notebook = ttk.Notebook(left)
        notebook.pack(fill=tk.BOTH, expand=True)

        tab_basic = ttk.Frame(notebook)
        tab_pid = ttk.Frame(notebook)
        tab_vision = ttk.Frame(notebook)
        tab_camera = ttk.Frame(notebook)
        notebook.add(tab_basic, text="基本")
        notebook.add(tab_pid, text="PID")
        notebook.add(tab_vision, text="视觉")
        notebook.add(tab_camera, text="相机")
        tab_basic = self._make_scrollable_tab(tab_basic)
        tab_pid = self._make_scrollable_tab(tab_pid)
        tab_vision = self._make_scrollable_tab(tab_vision)
        tab_camera = self._make_scrollable_tab(tab_camera)

        tab_basic.columnconfigure(1, weight=1)
        tab_basic.columnconfigure(3, weight=1)
        r = 0
        ttk.Label(tab_basic, text="控制方式").grid(row=r, column=0, sticky="w", padx=(0, 6), pady=(2, 2))
        ttk.Label(tab_basic, text="无刷RS485").grid(row=r, column=1, sticky="w", pady=(2, 2))
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
        ttk.Checkbutton(tab_basic, text="无刷并行写(提Hz/可能不稳)", variable=self.brushless_parallel_write).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 2))
        r += 1
        self._grid_entry(tab_basic, r, 0, "关机水平角", self.shutdown_pan_deg, width=8)
        self._grid_entry(tab_basic, r, 2, "关机俯仰角", self.shutdown_tilt_deg, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "关机速度dps", self.shutdown_speed_dps, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "控制周期ms", self.control_period_ms, width=8)
        self._grid_entry(tab_basic, r, 2, "稳定周期ms", self.stab_period_ms, width=8)
        r += 1
        ttk.Checkbutton(tab_basic, text="控制调试输出", variable=self.control_debug_timing).grid(row=r, column=0, sticky="w", pady=(2, 2))
        r += 1
        ttk.Checkbutton(tab_basic, text="绑定CPU核心", variable=self.core_affinity_enabled).grid(row=r, column=0, sticky="w", pady=(2, 2))
        self._grid_entry(tab_basic, r, 2, "UI核", self.core_ui, width=6)
        r += 1
        ttk.Combobox(
            tab_basic,
            textvariable=self.core_layout_preset,
            values=("标准版(UI0 视1-2 控3)", "更稳版(UI0 视1-2 控3 稳3)"),
            state="readonly",
            width=28,
        ).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 2))
        ttk.Button(tab_basic, text="应用核预设", command=self._apply_core_layout_preset).grid(row=r, column=2, columnspan=2, sticky="w", pady=(2, 2))
        r += 1
        self._grid_entry(tab_basic, r, 0, "控制核", self.core_control, width=6)
        self._grid_entry(tab_basic, r, 2, "视觉核(可1,2)", self.core_vision, width=8)
        r += 1
        self._grid_entry(tab_basic, r, 0, "稳定核", self.core_stabilize, width=6)
        r += 1
        self._grid_entry(tab_basic, r, 0, "点动角度", self.jog_step_deg, width=8)
        r += 1
        self._refresh_servo_mode_ui()
        ttk.Checkbutton(tab_basic, text="启用跟踪", variable=self.track_enabled).grid(row=r, column=0, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="启用水平", variable=self.pan_enabled).grid(row=r, column=1, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="启用俯仰", variable=self.tilt_enabled).grid(row=r, column=2, sticky="w", pady=(6, 0))
        ttk.Checkbutton(tab_basic, text="自动稳定", variable=self.auto_stabilize).grid(row=r, column=3, sticky="w", pady=(6, 0))
        r += 1
        ttk.Checkbutton(tab_basic, text="多进程模式(仅多进程)", variable=self.multiprocess_mode, state=tk.DISABLED).grid(row=r, column=0, columnspan=2, sticky="w", pady=(2, 0))
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
        ttk.Checkbutton(imu_frame, text="X反向(Yaw/水平)", variable=self.stab_invert_x).grid(row=2, column=2, sticky="w")
        ttk.Checkbutton(imu_frame, text="Y反向(Pitch/俯仰)", variable=self.stab_invert_y).grid(row=2, column=3, sticky="w")
        ttk.Checkbutton(imu_frame, text="稳定Pitch", variable=self.stab_enable_pitch).grid(row=3, column=0, columnspan=2, sticky="w", pady=(2, 4))
        ttk.Checkbutton(imu_frame, text="稳定Yaw", variable=self.stab_enable_yaw).grid(row=3, column=2, columnspan=2, sticky="w", pady=(2, 4))
        ttk.Label(imu_frame, text="Pitch增益:").grid(row=4, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_gain_pitch, width=10).grid(row=4, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw增益:").grid(row=4, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_gain_yaw, width=10).grid(row=4, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch死区(°):").grid(row=5, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pitch_deadband_deg, width=10).grid(row=5, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw死区(°):").grid(row=5, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_yaw_deadband_deg, width=10).grid(row=5, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch限幅(°):").grid(row=6, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_tilt_limit_deg, width=10).grid(row=6, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw限幅(°):").grid(row=6, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pan_limit_deg, width=10).grid(row=6, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch平滑α:").grid(row=7, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_tilt_alpha, width=10).grid(row=7, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw平滑α:").grid(row=7, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pan_alpha, width=10).grid(row=7, column=3, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Pitch速度限幅(°/s):").grid(row=8, column=0, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_tilt_rate_limit_deg_per_s, width=10).grid(row=8, column=1, sticky="w", padx=5)
        ttk.Label(imu_frame, text="Yaw速度限幅(°/s):").grid(row=8, column=2, sticky="e")
        ttk.Entry(imu_frame, textvariable=self.stab_pan_rate_limit_deg_per_s, width=10).grid(row=8, column=3, sticky="w", padx=5)
        ttk.Button(imu_frame, text="IMU置零", command=self._zero_imu).grid(row=9, column=0, sticky="ew", pady=(8, 0))
        ttk.Button(imu_frame, text="零偏设置", command=self._open_imu_offsets_dialog).grid(row=9, column=1, sticky="ew", pady=(8, 0), padx=(5, 0))
        ttk.Label(imu_frame, text="Pitch:").grid(row=9, column=2, sticky="e", pady=(8, 0))
        ttk.Label(imu_frame, textvariable=self.imu_status_pitch).grid(row=9, column=3, sticky="w", pady=(8, 0))
        ttk.Label(imu_frame, text="Yaw:").grid(row=10, column=2, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_yaw).grid(row=10, column=3, sticky="w")
        ttk.Label(imu_frame, text="Age:").grid(row=10, column=0, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_age).grid(row=10, column=1, sticky="w")
        ttk.Label(imu_frame, text="基准Pitch:").grid(row=11, column=0, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_pitch_base).grid(row=11, column=1, sticky="w")
        ttk.Label(imu_frame, text="基准Yaw:").grid(row=11, column=2, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_yaw_base).grid(row=11, column=3, sticky="w")
        ttk.Label(imu_frame, text="ΔPitch:").grid(row=12, column=0, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_pitch_delta).grid(row=12, column=1, sticky="w")
        ttk.Label(imu_frame, text="ΔYaw:").grid(row=12, column=2, sticky="e")
        ttk.Label(imu_frame, textvariable=self.imu_status_yaw_delta).grid(row=12, column=3, sticky="w")

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
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kP", self.kp_x, 0.0, 0.01, range_editable=True)
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kI", self.ki_x, 0.0, 0.01, range_editable=True)
        row_x = self._grid_slider(pid_x_frame, row_x, 0, "kD", self.kd_x, 0.0, 0.01, range_editable=True)
        row_y = 0
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kP", self.kp_y, 0.0, 0.01, range_editable=True)
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kI", self.ki_y, 0.0, 0.01, range_editable=True)
        row_y = self._grid_slider(pid_y_frame, row_y, 0, "kD", self.kd_y, 0.0, 0.01, range_editable=True)
        common = ttk.LabelFrame(tab_pid, text="通用", padding=8)
        common.pack(fill=tk.X, pady=(10, 0))
        common.columnconfigure(0, weight=1)
        common.columnconfigure(1, weight=1)
        r2 = 0
        r2 = self._grid_slider(common, r2, 0, "死区(像素)", self.error_deadband, 0.0, 30.0)
        ttk.Label(common, text="误差小于此值时不响应，避免抖动", font=("", 8), foreground="gray").grid(row=r2-1, column=2, sticky="w", padx=(6, 0))

        # 卡尔曼滤波参数配置
        kalman_frame = ttk.LabelFrame(tab_pid, text="卡尔曼滤波 (Kalman Filter)", padding=8)
        kalman_frame.pack(fill=tk.X, pady=(10, 0))
        kalman_frame.columnconfigure(0, weight=1)
        kalman_frame.columnconfigure(1, weight=1)
        rk = 0
        ttk.Checkbutton(kalman_frame, text="启用卡尔曼滤波", variable=self.kalman_enabled).grid(row=rk, column=0, columnspan=2, sticky="w", pady=(2, 6))
        rk += 1
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
        r3 = self._grid_slider(servo_range, r3, 0, "水平最小", self.pan_min, -360.0, 0.0, colspan=1)
        # 上一行的调用返回的是 r3+1，为了让最大和最小在同一行，我们需要把行号退回
        self._grid_slider(servo_range, r3-1, 1, "水平最大", self.pan_max, 0.0, 360.0, colspan=1)
        r3 = self._grid_slider(servo_range, r3, 0, "俯仰最小", self.tilt_min, -360.0, 0.0, colspan=1)
        self._grid_slider(servo_range, r3-1, 1, "俯仰最大", self.tilt_max, 0.0, 360.0, colspan=1)
        entry_row = ttk.Frame(servo_range)
        entry_row.grid(row=r3, column=0, columnspan=2, sticky="ew", pady=(2, 4))
        ttk.Label(entry_row, text="水平最小").grid(row=0, column=0, sticky="e")
        ttk.Entry(entry_row, textvariable=self.pan_min, width=8).grid(row=0, column=1, sticky="w", padx=(4, 10))
        ttk.Label(entry_row, text="水平最大").grid(row=0, column=2, sticky="e")
        ttk.Entry(entry_row, textvariable=self.pan_max, width=8).grid(row=0, column=3, sticky="w", padx=(4, 10))
        ttk.Label(entry_row, text="俯仰最小").grid(row=0, column=4, sticky="e")
        ttk.Entry(entry_row, textvariable=self.tilt_min, width=8).grid(row=0, column=5, sticky="w", padx=(4, 10))
        ttk.Label(entry_row, text="俯仰最大").grid(row=0, column=6, sticky="e")
        ttk.Entry(entry_row, textvariable=self.tilt_max, width=8).grid(row=0, column=7, sticky="w", padx=(4, 0))
        r3 += 1
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
        hw_range = ttk.LabelFrame(tab_pid, text="舵机物理边界", padding=8)
        hw_range.pack(fill=tk.X, pady=(10, 0))
        hw_range.columnconfigure(0, weight=1)
        hw_range.columnconfigure(1, weight=1)
        
        self.str_hw_pan_min = tk.StringVar(value="-360.0°")
        self.str_hw_pan_max = tk.StringVar(value="360.0°")
        self.str_hw_tilt_min = tk.StringVar(value="-360.0°")
        self.str_hw_tilt_max = tk.StringVar(value="360.0°")

        ttk.Label(hw_range, text="水平最小:").grid(row=0, column=0, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_pan_min).grid(row=0, column=1, sticky="w", padx=5)
        ttk.Label(hw_range, text="水平最大:").grid(row=0, column=2, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_pan_max).grid(row=0, column=3, sticky="w", padx=5)
        ttk.Label(hw_range, text="俯仰最小:").grid(row=1, column=0, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_tilt_min).grid(row=1, column=1, sticky="w", padx=5)
        ttk.Label(hw_range, text="俯仰最大:").grid(row=1, column=2, sticky="e")
        ttk.Label(hw_range, textvariable=self.str_hw_tilt_max).grid(row=1, column=3, sticky="w", padx=5)
        ttk.Label(hw_range, text="无刷电机物理边界固定为 ±360°（不做硬件读取）", foreground="gray").grid(row=2, column=0, columnspan=4, sticky="w", pady=(4, 0))
        
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
            return left + (float(v) + 360.0) / 720.0 * (right - left)

        def draw_one(canvas, gui_min, gui_max, hw_min, hw_max):
            w = max(120, canvas.winfo_width())
            h = max(20, canvas.winfo_height())
            y = h / 2.0
            canvas.delete("all")
            x0 = map_x(-360.0, w)
            x1 = map_x(360.0, w)
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
        self._grid_entry(left_vis, rv, 0, "识别裁切宽", self.hough_crop_width, width=8)
        rv += 1
        self._grid_entry(left_vis, rv, 0, "识别裁切高", self.hough_crop_height, width=8)
        rv += 1
        rv = self._grid_slider(left_vis, rv, 0, "最小间距", self.min_dist, 10, 300)
        rv = self._grid_slider(left_vis, rv, 0, "参数1", self.param1, 50, 500)
        rv = self._grid_slider(left_vis, rv, 0, "参数2", self.param2, 5, 200)
        rv = self._grid_slider(left_vis, rv, 0, "最小半径", self.min_radius, 1, 300)
        rv = self._grid_slider(left_vis, rv, 0, "最大半径", self.max_radius, 1, 300)
        ttk.Checkbutton(left_vis, text="启用EMA平滑", variable=self.detect_ema_enabled).grid(row=rv, column=0, columnspan=2, sticky="w", pady=(2, 6))
        rv += 1
        rv = self._grid_slider(left_vis, rv, 0, "EMA系数α", self.detect_ema_alpha, 0.01, 1.0)
        rv2 = 0
        rv2 = self._grid_slider(right_vis, rv2, 0, "模糊核大小", self.ksize, 3, 19)
        ttk.Checkbutton(right_vis, text="显示四宫格调试(更耗性能)", variable=self.show_debug_panels).grid(row=rv2, column=0, columnspan=2, sticky="w", pady=(2, 6))
        rv2 += 1
        
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
        w_entry = self._grid_entry(cam, rc, 0, "采集宽度", self.camera_raw_width, width=10, return_widget=True)
        rc += 1
        h_entry = self._grid_entry(cam, rc, 0, "采集高度", self.camera_raw_height, width=10, return_widget=True)
        rc += 1
        self._grid_entry(cam, rc, 0, "传感器BitDepth(10/12)", self.sensor_bit_depth, width=10)
        rc += 1
        rc = self._grid_slider(cam, rc, 0, "视频裁切比例", self.video_crop_ratio, 0.2, 1.0)
        ttk.Button(cam, text="应用分辨率", command=self._apply_camera_resolution).grid(row=rc, column=0, sticky="w", pady=(2, 8))
        def _auto_apply_resolution(_event=None):
            try:
                self._apply_camera_resolution()
            except Exception:
                pass
        try:
            if w_entry is not None:
                w_entry.bind("<Return>", _auto_apply_resolution)
                w_entry.bind("<FocusOut>", _auto_apply_resolution)
            if h_entry is not None:
                h_entry.bind("<Return>", _auto_apply_resolution)
                h_entry.bind("<FocusOut>", _auto_apply_resolution)
        except Exception:
            pass
        rc += 1
        rc = self._grid_slider(cam, rc, 0, "相机FPS", self.camera_fps, 10, 120)
        rc = self._grid_slider(cam, rc, 0, "GUI刷新Hz", self.ui_refresh_hz, 5, 60)
        rc = self._grid_slider(cam, rc, 0, "多进程预览Hz", self.preview_push_hz, 2, 30)
        rc = self._grid_slider(cam, rc, 0, "显示刷新Hz", self.preview_render_hz, 2, 60)
        rotate_frame = ttk.Frame(cam)
        rotate_frame.grid(row=rc, column=0, sticky="ew", pady=(2, 6))
        rotate_frame.columnconfigure(0, weight=1)
        rotate_header = ttk.Frame(rotate_frame)
        rotate_header.grid(row=0, column=0, sticky="ew")
        rotate_header.columnconfigure(0, weight=1)
        ttk.Label(rotate_header, text="图像旋转(°)").grid(row=0, column=0, sticky="w")
        rotate_value_text = tk.StringVar(value=f"{self.image_rotate_deg.get():.2f}")
        ttk.Label(rotate_header, textvariable=rotate_value_text).grid(row=0, column=1, sticky="e")
        rotate_scale = tk.Scale(
            rotate_frame,
            from_=-180.0,
            to=180.0,
            orient=tk.HORIZONTAL,
            resolution=0.1,
            variable=self.image_rotate_deg,
            showvalue=False,
            length=380,
        )
        rotate_scale.grid(row=1, column=0, sticky="ew")
        ttk.Entry(rotate_frame, textvariable=self.image_rotate_deg, width=10).grid(row=2, column=0, sticky="w", pady=(2, 0))
        self.image_rotate_deg.trace_add("write", lambda *_: rotate_value_text.set(f"{self.image_rotate_deg.get():.2f}"))
        rc += 1
        
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

    def _grid_entry(self, parent, row, col, text, var, width=10, return_widget=False):
        ttk.Label(parent, text=text).grid(row=row, column=col, sticky="w", padx=(0, 6), pady=(2, 2))
        entry = ttk.Entry(parent, textvariable=var, width=width)
        entry.grid(row=row, column=col + 1, sticky="ew", pady=(2, 2))
        if return_widget:
            return entry

    def _grid_slider(self, parent, row, col, text, var, low, high, colspan=2, range_editable=False):
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

        low_val = float(low)
        high_val = float(high)

        def _on_change(*_):
            if isinstance(var, tk.IntVar):
                value_text.set(str(int(var.get())))
            else:
                value_text.set(f"{var.get():.4f}")

        var.trace_add("write", _on_change)

        # 添加键盘方向键支持
        # 为了让小数步进更精细，整数步进保持1
        def _calc_steps():
            if isinstance(var, tk.IntVar):
                step = 1
                big_step = max(1, int((high_val - low_val) / 10.0))
            else:
                step = (high_val - low_val) / 200.0
                big_step = (high_val - low_val) / 20.0
            return step, big_step

        step, big_step = _calc_steps()

        def _on_key(event):
            current = var.get()
            if event.keysym == "Left":
                new_val = max(low_val, current - step)
            elif event.keysym == "Right":
                new_val = min(high_val, current + step)
            elif event.keysym == "Down":
                new_val = max(low_val, current - big_step)
            elif event.keysym == "Up":
                new_val = min(high_val, current + big_step)
            elif event.keysym == "Home":
                new_val = low_val
            elif event.keysym == "End":
                new_val = high_val
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

        if range_editable:
            range_row = ttk.Frame(frame)
            range_row.grid(row=2, column=0, sticky="ew", pady=(4, 0))
            range_row.columnconfigure(1, weight=1)
            range_row.columnconfigure(3, weight=1)
            min_text = tk.StringVar(value=str(low_val))
            max_text = tk.StringVar(value=str(high_val))
            ttk.Label(range_row, text="min").grid(row=0, column=0, sticky="w")
            e_min = ttk.Entry(range_row, textvariable=min_text, width=8)
            e_min.grid(row=0, column=1, sticky="w", padx=(4, 12))
            ttk.Label(range_row, text="max").grid(row=0, column=2, sticky="w")
            e_max = ttk.Entry(range_row, textvariable=max_text, width=8)
            e_max.grid(row=0, column=3, sticky="w", padx=(4, 0))

            def _apply_range(*_):
                nonlocal low_val, high_val, step, big_step
                try:
                    a = float(min_text.get())
                    b = float(max_text.get())
                except Exception:
                    return
                if b == a:
                    return
                if b < a:
                    a, b = b, a
                low_val = a
                high_val = b
                scale.configure(from_=low_val, to=high_val)
                v = float(var.get())
                if v < low_val:
                    var.set(low_val)
                elif v > high_val:
                    var.set(high_val)
                step, big_step = _calc_steps()

            e_min.bind("<Return>", _apply_range)
            e_max.bind("<Return>", _apply_range)
            e_min.bind("<FocusOut>", _apply_range)
            e_max.bind("<FocusOut>", _apply_range)

        return row + 1

    def _bind_current_thread_core(self, core_index):
        try:
            if not self.core_affinity_enabled.get():
                return False
            if os.name != "posix":
                return False
            if not hasattr(os, "sched_setaffinity"):
                return False
            if isinstance(core_index, str):
                spec = core_index.strip()
            else:
                spec = str(core_index).strip()
            if not spec:
                return False
            cores = set()
            for part in spec.split(","):
                p = part.strip()
                if not p:
                    continue
                if "-" in p:
                    a, b = p.split("-", 1)
                    start = max(0, int(a))
                    end = max(0, int(b))
                    if end < start:
                        start, end = end, start
                    cores.update(range(start, end + 1))
                else:
                    cores.add(max(0, int(p)))
            if not cores:
                return False
            tid = threading.get_native_id() if hasattr(threading, "get_native_id") else os.getpid()
            os.sched_setaffinity(tid, cores)
            return True
        except Exception:
            return False

    def _apply_core_layout_preset(self):
        p = self.core_layout_preset.get()
        if p == "标准版(UI0 视1-2 控3)":
            self.core_affinity_enabled.set(True)
            self.core_ui.set(0)
            self.core_control.set(3)
            self.core_vision.set("1,2")
            self.core_stabilize.set(3)
        elif p == "更稳版(UI0 视1-2 控3 稳3)":
            self.core_affinity_enabled.set(True)
            self.core_ui.set(0)
            self.core_control.set(3)
            self.core_vision.set("1,2")
            self.core_stabilize.set(3)
        self.status_text.set("已应用CPU核预设")

    def _start_multiprocess_runtime(self):
        self.mp_stop_event = self.mp_ctx.Event()
        self.mp_vision_settings_queue = self.mp_ctx.Queue(maxsize=4)
        self.mp_control_settings_queue = self.mp_ctx.Queue(maxsize=4)
        self.mp_control_cmd_queue = self.mp_ctx.Queue(maxsize=16)
        self.mp_status_queue = self.mp_ctx.Queue(maxsize=32)
        self.mp_latest_detection = self.mp_ctx.Array("d", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.mp_vision_proc = self.mp_ctx.Process(
            target=_vision_process_main,
            args=(self.mp_stop_event, self.mp_vision_settings_queue, self.mp_status_queue, self.mp_latest_detection),
            daemon=True,
        )
        self.mp_control_proc = self.mp_ctx.Process(
            target=_control_process_main,
            args=(self.mp_stop_event, self.mp_control_settings_queue, self.mp_control_cmd_queue, self.mp_status_queue, self.mp_latest_detection),
            daemon=True,
        )
        self.mp_vision_proc.start()
        self.mp_control_proc.start()
        settings = self._get_settings()
        settings["tracking_active"] = bool(self.tracking_active)
        settings["multiprocess_preview"] = bool(self.multiprocess_preview.get())
        try:
            self.mp_vision_settings_queue.put_nowait(settings)
            self.mp_control_settings_queue.put_nowait(settings)
        except Exception:
            pass

    def _stop_multiprocess_runtime(self):
        if self.mp_stop_event is not None:
            try:
                self.mp_stop_event.set()
            except Exception:
                pass
        for p in (self.mp_vision_proc, self.mp_control_proc):
            if p is not None:
                try:
                    p.join(timeout=1.0)
                except Exception:
                    pass
                try:
                    if p.is_alive():
                        p.terminate()
                except Exception:
                    pass
        self.mp_vision_proc = None
        self.mp_control_proc = None
        self.mp_stop_event = None
        self.mp_vision_settings_queue = None
        self.mp_control_settings_queue = None
        self.mp_control_cmd_queue = None
        self.mp_status_queue = None
        self.mp_latest_detection = None
        self.mp_preview_jpg = None

    def _push_multiprocess_settings(self):
        if self.mp_vision_settings_queue is None or self.mp_control_settings_queue is None:
            return
        s = self._get_settings()
        s["tracking_active"] = bool(self.tracking_active)
        s["multiprocess_preview"] = bool(self.multiprocess_preview.get())
        for q in (self.mp_vision_settings_queue, self.mp_control_settings_queue):
            try:
                if q.full():
                    q.get_nowait()
                q.put_nowait(s)
            except Exception:
                pass

    def _start_runtime(self):
        if self.running:
            return
        try:
            self.multiprocess_mode.set(True)
            self.aggressive_perf_mode.set(False)
            self.multiprocess_preview.set(True)
            self.stop_event.clear()
            self.detect_stop_event.clear()
            self.stab_stop_event.clear()
            self.camera_reconfigure_event.clear()
            self.worker_error = None
            self.mp_last_error = ""
            self._update_settings_from_vars()
            self._bind_current_thread_core(self.core_ui.get())
            self._release_local_hardware_handles()
            self._start_multiprocess_runtime()
            self.running = True
            self.after_id = self.root.after(self._ui_refresh_delay_ms(), self._ui_loop)
            self.status_text.set("多进程模式运行中")
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
            if self.multiprocess_mode.get():
                self.status_text.set("多进程运行中（未跟踪）")
            else:
                self.status_text.set("检测中（未跟踪）")
        # 停止时不再回正，直接原地保持

    def reset_axes(self):
        self.pid_x.reset()
        self.pid_y.reset()
        self.kalman = Kalman2D()
        if self.mp_control_cmd_queue is not None:
            try:
                self.mp_control_cmd_queue.put_nowait(("recenter",))
            except Exception:
                pass
        with self.detect_lock:
            self.latest_detection = None
            self.latest_detection_time = 0.0
        
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

    def _move_to_shutdown_pose(self):
        if self.servo is None:
            return
        s = self._get_settings()
        pan_target = float(s.get("shutdown_pan_deg", 0.0))
        tilt_target = float(s.get("shutdown_tilt_deg", 0.0))
        shutdown_speed = max(1.0, float(s.get("shutdown_speed_dps", 60.0)))
        pan_backup = getattr(self.servo, "pan_speed_dps", None)
        tilt_backup = getattr(self.servo, "tilt_speed_dps", None)
        try:
            if pan_backup is not None:
                self.servo.pan_speed_dps = shutdown_speed
            if tilt_backup is not None:
                self.servo.tilt_speed_dps = shutdown_speed
            self.servo.set_angles(
                [
                    (self.active_pan_id, pan_target),
                    (self.active_tilt_id, tilt_target),
                ]
            )
            self.servo.move_angle(wait=False)
            self.current_pan_angle = pan_target
            self.current_tilt_angle = tilt_target
            time.sleep(0.35)
        finally:
            if pan_backup is not None:
                self.servo.pan_speed_dps = pan_backup
            if tilt_backup is not None:
                self.servo.tilt_speed_dps = tilt_backup

    def _worker_loop(self):
        self._bind_current_thread_core(self.core_control.get())
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
                if self.camera_reconfigure_event.is_set():
                    self.camera_reconfigure_event.clear()
                    try:
                        if self.picam2 is not None:
                            self.picam2.stop()
                            self.picam2.close()
                            self.picam2 = None
                        self._ensure_camera()
                        if getattr(self, "camera_raw_size", None) is not None:
                            rw, rh = self.camera_raw_size
                            self.status_text.set(f"相机分辨率已应用: {rw}x{rh}")
                    except Exception as exc:
                        self.worker_error = str(exc)
                        self.stop_event.set()
                        break
                    continue
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
                try:
                    x0, y0, cw, ch = _center_crop_rect(frame_rgb.shape[1], frame_rgb.shape[0], s.get("video_crop_ratio", 1.0))
                    frame_rgb = frame_rgb[y0 : y0 + ch, x0 : x0 + cw]
                except Exception:
                    pass
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

                if bool(s.get("kalman_enabled", True)):
                    self.kalman.update_params(s["kalman_process_noise"], s["kalman_measurement_noise"])
                    kalman_out = self.kalman.update(measurement)
                    if kalman_out is None:
                        filtered_x, filtered_y = float(center_x), float(center_y)
                        pred_x, pred_y = float(center_x), float(center_y)
                    else:
                        filtered_x, filtered_y, pred_x, pred_y = kalman_out
                else:
                    if measurement is None:
                        filtered_x, filtered_y = float(center_x), float(center_y)
                    else:
                        filtered_x, filtered_y = measurement
                    pred_x, pred_y = filtered_x, filtered_y
                    
                # 目标圆心坐标（这里暂不加 bias，后面根据对准模式决定）
                target_x = filtered_x
                target_y = filtered_y

                if not s["track_enabled"]:
                    target_x = float(center_x)
                    target_y = float(center_y)

                self.pid_x.set_gains(s["kp_x"], s["ki_x"], s["kd_x"])
                self.pid_y.set_gains(s["kp_y"], s["ki_y"], s["kd_y"])
                
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

                do_track = self.tracking_active
                do_stab = bool(s.get("auto_stabilize", False))
                if do_track:
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
                    with self.motion_lock:
                        stab_pan = float(self.stab_pan_comp_deg)
                        stab_tilt = float(self.stab_tilt_comp_deg)
                # 使用GUI配置和硬件物理边界的交集作为最终限制
                # max(硬件最小, GUI最小) 和 min(硬件最大, GUI最大)
                pan_min = max(float(s.get("hw_pan_min", -360.0)), float(s.get("pan_min", -360.0)))
                pan_max = min(float(s.get("hw_pan_max", 360.0)), float(s.get("pan_max", 360.0)))
                tilt_min = max(float(s.get("hw_tilt_min", -360.0)), float(s.get("tilt_min", -360.0)))
                tilt_max = min(float(s.get("hw_tilt_max", 360.0)), float(s.get("tilt_max", 360.0)))
                gui_pan_min = float(s.get("pan_min", -360.0))
                gui_pan_max = float(s.get("pan_max", 360.0))
                gui_tilt_min = float(s.get("tilt_min", -360.0))
                gui_tilt_max = float(s.get("tilt_max", 360.0))
                hw_pan_min = float(s.get("hw_pan_min", -360.0))
                hw_pan_max = float(s.get("hw_pan_max", 360.0))
                hw_tilt_min = float(s.get("hw_tilt_min", -360.0))
                hw_tilt_max = float(s.get("hw_tilt_max", 360.0))

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
                    self.current_pan_angle = max(pan_min, min(pan_max, self.current_pan_angle + delta_x))

                if do_track and s["tilt_enabled"]:
                    delta_y = self.pid_y.update(error_y, dt=dt)
                    # 边界限制与抗积分饱和 (Anti-windup)
                    if tilt_at_min_before and delta_y < 0:
                        delta_y = 0
                        self.pid_y.i_term = 0.0
                    if tilt_at_max_before and delta_y > 0:
                        delta_y = 0
                        self.pid_y.i_term = 0.0
                    self.current_tilt_angle = max(tilt_min, min(tilt_max, self.current_tilt_angle + delta_y))

                # 更新后的跟踪输出边界（不含IMU补偿）
                pan_at_min = self.current_pan_angle <= pan_min
                pan_at_max = self.current_pan_angle >= pan_max
                tilt_at_min = self.current_tilt_angle <= tilt_min
                tilt_at_max = self.current_tilt_angle >= tilt_max
                bound_reason_msgs = []

                if do_track and (s["pan_enabled"] or s["tilt_enabled"]):
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

    def _stabilization_loop(self):
        self._bind_current_thread_core(self.core_stabilize.get())
        last_time = time.time()
        last_auto_stabilize = False
        try:
            while not self.stab_stop_event.is_set():
                if not self.running:
                    time.sleep(0.01)
                    last_time = time.time()
                    continue
                s = self._get_settings()
                period_ms = max(2, int(s.get("stab_period_ms", 12)))
                loop_start = time.time()
                dt = max(loop_start - last_time, 1e-4)
                last_time = loop_start
                auto_stabilize = bool(s.get("auto_stabilize", False))
                if not auto_stabilize:
                    with self.motion_lock:
                        self.stab_pan_comp_deg = 0.0
                        self.stab_tilt_comp_deg = 0.0
                    last_auto_stabilize = False
                    time.sleep(period_ms / 1000.0)
                    continue
                try:
                    self._ensure_servo()
                    self._ensure_imu()
                    imu_state = self.imu.get_state() if self.imu is not None else None
                except Exception as exc:
                    self.worker_error = f"IMU stabilize failed: {exc}"
                    self.stop_event.set()
                    break
                if imu_state is None or imu_state.last_update <= 0:
                    time.sleep(period_ms / 1000.0)
                    continue
                if not last_auto_stabilize:
                    with self.motion_lock:
                        self.imu_zero_pitch = float(imu_state.pitch_deg)
                        self.imu_zero_yaw = float(imu_state.yaw_deg)
                        self.stab_pan_residual_deg = 0.0
                        self.stab_tilt_residual_deg = 0.0
                        self.stab_pan_filtered_deg = 0.0
                        self.stab_tilt_filtered_deg = 0.0
                        self.stab_pan_comp_deg = 0.0
                        self.stab_tilt_comp_deg = 0.0
                    last_auto_stabilize = True
                pitch_err = self._angle_diff_deg(imu_state.pitch_deg, self.imu_zero_pitch)
                yaw_err = self._angle_diff_deg(imu_state.yaw_deg, self.imu_zero_yaw)
                enable_pitch = bool(s.get("stab_enable_pitch", True))
                enable_yaw = bool(s.get("stab_enable_yaw", False))
                if bool(s.get("stab_invert_y", False)):
                    pitch_err = -pitch_err
                if bool(s.get("stab_invert_x", False)):
                    yaw_err = -yaw_err
                if not enable_pitch:
                    pitch_err = 0.0
                if not enable_yaw:
                    yaw_err = 0.0
                pitch_deadband = max(0.0, float(s.get("stab_pitch_deadband_deg", 0.6)))
                if abs(pitch_err) < pitch_deadband:
                    pitch_err = 0.0
                else:
                    pitch_err = math.copysign(abs(pitch_err) - pitch_deadband, pitch_err)
                yaw_deadband = max(0.0, float(s.get("stab_yaw_deadband_deg", 0.6)))
                if abs(yaw_err) < yaw_deadband:
                    yaw_err = 0.0
                else:
                    yaw_err = math.copysign(abs(yaw_err) - yaw_deadband, yaw_err)
                tilt_limit = max(0.0, float(s.get("stab_tilt_limit_deg", 8.0)))
                pan_limit = max(0.0, float(s.get("stab_pan_limit_deg", 8.0)))
                stab_tilt_target = pitch_err * float(s.get("stab_gain_pitch", 1.0))
                stab_tilt_target = max(-tilt_limit, min(tilt_limit, stab_tilt_target))
                stab_pan_target = -yaw_err * float(s.get("stab_gain_yaw", 1.0))
                stab_pan_target = max(-pan_limit, min(pan_limit, stab_pan_target))
                tilt_alpha = max(0.0, min(1.0, float(s.get("stab_tilt_alpha", 0.35))))
                pan_alpha = max(0.0, min(1.0, float(s.get("stab_pan_alpha", 0.35))))
                tilt_filtered_target = self.stab_tilt_filtered_deg + tilt_alpha * (stab_tilt_target - self.stab_tilt_filtered_deg)
                pan_filtered_target = self.stab_pan_filtered_deg + pan_alpha * (stab_pan_target - self.stab_pan_filtered_deg)
                tilt_rate_limit = max(0.0, float(s.get("stab_tilt_rate_limit_deg_per_s", 120.0)))
                pan_rate_limit = max(0.0, float(s.get("stab_pan_rate_limit_deg_per_s", 120.0)))
                tilt_delta_filtered = tilt_filtered_target - self.stab_tilt_filtered_deg
                pan_delta_filtered = pan_filtered_target - self.stab_pan_filtered_deg
                tilt_max_delta = tilt_rate_limit * dt
                pan_max_delta = pan_rate_limit * dt
                if tilt_delta_filtered > tilt_max_delta:
                    tilt_delta_filtered = tilt_max_delta
                elif tilt_delta_filtered < -tilt_max_delta:
                    tilt_delta_filtered = -tilt_max_delta
                if pan_delta_filtered > pan_max_delta:
                    pan_delta_filtered = pan_max_delta
                elif pan_delta_filtered < -pan_max_delta:
                    pan_delta_filtered = -pan_max_delta
                self.stab_tilt_filtered_deg = self.stab_tilt_filtered_deg + tilt_delta_filtered
                self.stab_pan_filtered_deg = self.stab_pan_filtered_deg + pan_delta_filtered
                stab_tilt_raw = float(self.stab_tilt_filtered_deg)
                stab_pan_raw = float(self.stab_pan_filtered_deg)
                if s.get("servo_mode") == "无刷RS485":
                    stab_tilt = stab_tilt_raw
                    stab_pan = stab_pan_raw
                else:
                    stab_tilt = self._quantize_to_servo_step_deg(stab_tilt_raw, axis="tilt")
                    stab_pan = self._quantize_to_servo_step_deg(stab_pan_raw, axis="pan")
                with self.motion_lock:
                    self.stab_pan_comp_deg = float(stab_pan)
                    self.stab_tilt_comp_deg = float(stab_tilt)
                    self.latest_imu = (
                        float(imu_state.pitch_deg),
                        float(imu_state.yaw_deg),
                        float(time.time() - imu_state.last_update),
                        float(pitch_err),
                        float(yaw_err),
                    )
                if not self.tracking_active and self.servo is not None:
                    pan_min = max(float(s.get("hw_pan_min", -360.0)), float(s.get("pan_min", -360.0)))
                    pan_max = min(float(s.get("hw_pan_max", 360.0)), float(s.get("pan_max", 360.0)))
                    tilt_min = max(float(s.get("hw_tilt_min", -360.0)), float(s.get("tilt_min", -360.0)))
                    tilt_max = min(float(s.get("hw_tilt_max", 360.0)), float(s.get("tilt_max", 360.0)))
                    with self.motion_lock:
                        out_pan = float(self.current_pan_angle)
                        out_tilt = float(self.current_tilt_angle)
                    if s.get("pan_enabled", True):
                        out_pan = max(pan_min, min(pan_max, out_pan + float(stab_pan)))
                    if s.get("tilt_enabled", True):
                        out_tilt = max(tilt_min, min(tilt_max, out_tilt + float(stab_tilt)))
                    try:
                        self.servo.set_angles(
                            [
                                (self.active_pan_id, out_pan),
                                (self.active_tilt_id, out_tilt),
                            ]
                        )
                        self.servo.move_angle(wait=False)
                    except Exception:
                        pass
                elapsed_ms = int((time.time() - loop_start) * 1000)
                sleep_ms = max(0, period_ms - elapsed_ms)
                if sleep_ms > 0:
                    time.sleep(sleep_ms / 1000.0)
        except Exception as exc:
            self.worker_error = str(exc)
            self.stop_event.set()

    def _detect_loop(self):
        self._bind_current_thread_core(self.core_vision.get())
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
                h_full, w_full = frame_rgb.shape[:2]
                crop_w = max(0, int(s.get("hough_crop_width", w_full)))
                crop_h = max(0, int(s.get("hough_crop_height", h_full)))
                if crop_w <= 0:
                    crop_w = w_full
                if crop_h <= 0:
                    crop_h = h_full
                crop_w = min(crop_w, w_full)
                crop_h = min(crop_h, h_full)
                with self.detect_lock:
                    last_det = self.latest_detection
                    last_det_time = self.latest_detection_time
                crop_x0 = 0
                crop_y0 = 0
                if last_det is not None and (time.time() - last_det_time) <= self.detect_stale_sec:
                    det_cx = int(last_det[0])
                    det_cy = int(last_det[1])
                    crop_x0 = max(0, min(w_full - crop_w, det_cx - (crop_w // 2)))
                    crop_y0 = max(0, min(h_full - crop_h, det_cy - (crop_h // 2)))
                    frame_for_detect = frame_rgb[crop_y0:crop_y0 + crop_h, crop_x0:crop_x0 + crop_w]
                else:
                    frame_for_detect = frame_rgb
                roi = None
                detection, blurred_green, blurred_red, offset_x, offset_y, scale = self._detect_circle(
                    frame_for_detect,
                    ksize=s["ksize"],
                    min_dist=s["min_dist"],
                    param1=s["param1"],
                    param2=s["param2"],
                    min_radius=s["min_radius"],
                    max_radius=s["max_radius"],
                    roi=roi,
                )
                if detection is not None:
                    detection = (detection[0] + crop_x0, detection[1] + crop_y0, detection[2])
                
                # 应用EMA低通滤波稳定检测结果
                if detection is not None:
                    if bool(s.get("detect_ema_enabled", True)):
                        if self.smoothed_detection is None:
                            self.smoothed_detection = list(detection)
                        else:
                            alpha = max(0.01, min(1.0, float(s.get("detect_ema_alpha", 0.3))))
                            self.smoothed_detection[0] = alpha * detection[0] + (1 - alpha) * self.smoothed_detection[0]
                            self.smoothed_detection[1] = alpha * detection[1] + (1 - alpha) * self.smoothed_detection[1]
                            self.smoothed_detection[2] = alpha * detection[2] + (1 - alpha) * self.smoothed_detection[2]
                        detection_to_save = tuple(self.smoothed_detection)
                    else:
                        self.smoothed_detection = None
                        detection_to_save = detection
                else:
                    self.smoothed_detection = None
                    detection_to_save = None

                with self.detect_lock:
                    self.latest_detection = detection_to_save
                    self.latest_detection_time = time.time()
                    self.latest_green_channel = (blurred_green, blurred_red, offset_x + crop_x0, offset_y + crop_y0, scale)
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
        if self.worker_error is not None and not self.multiprocess_mode.get():
            self.status_text.set(f"工作线程错误: {self.worker_error}")
            self.after_id = self.root.after(200, self._ui_loop)
            return
        if self.multiprocess_mode.get():
            self._push_multiprocess_settings()
            vision_alive = bool(self.mp_vision_proc is not None and self.mp_vision_proc.is_alive())
            control_alive = bool(self.mp_control_proc is not None and self.mp_control_proc.is_alive())
            if not vision_alive:
                self.mp_last_error = "视觉进程已退出"
            if not control_alive:
                self.mp_last_error = "控制进程已退出"
            try:
                while True:
                    key, val = self.mp_status_queue.get_nowait()
                    if key == "vision_hz":
                        self.mp_vision_hz = float(val)
                    elif key == "control_hz":
                        self.mp_control_hz = float(val)
                    elif key == "det_age":
                        self.mp_det_age = float(val)
                    elif key == "det_valid":
                        self.mp_det_valid = float(val)
                    elif key == "det_update_age":
                        self.mp_det_update_age = float(val)
                    elif key == "ctrl_timing":
                        try:
                            self.mp_ctrl_timing = (
                                f"cmd={float(val.get('cmd_ms', 0.0)):.1f}ms "
                                f"set={float(val.get('settings_ms', 0.0)):.1f}ms "
                                f"servo={float(val.get('servo_ms', 0.0)):.1f}ms "
                                f"imuI={float(val.get('imu_init_ms', 0.0)):.1f}ms "
                                f"imuR={float(val.get('imu_read_ms', 0.0)):.1f}ms "
                                f"write={float(val.get('servo_write_ms', 0.0)):.1f}ms "
                                f"work={float(val.get('work_ms', 0.0)):.1f}ms "
                                f"sleep={float(val.get('sleep_ms', 0.0)):.1f}ms"
                            )
                        except Exception:
                            self.mp_ctrl_timing = ""
                    elif key == "vision_err":
                        self.mp_last_error = f"视觉进程错误: {val}"
                    elif key == "control_err":
                        self.mp_last_error = f"控制进程错误: {val}"
                    elif key == "control_pan":
                        self.mp_pan = float(val)
                    elif key == "control_tilt":
                        self.mp_tilt = float(val)
                    elif key == "imu_pitch":
                        self.imu_status_pitch.set(f"{float(val):+.2f}")
                    elif key == "imu_yaw":
                        self.imu_status_yaw.set(f"{float(val):+.2f}")
                    elif key == "imu_age":
                        if float(val) >= 0:
                            self.imu_status_age.set(f"{float(val):.2f}s")
                        else:
                            self.imu_status_age.set("-")
                    elif key == "imu_zero_base":
                        self.imu_zero_pitch = float(val[0])
                        self.imu_zero_yaw = float(val[1])
                        self.imu_status_pitch_base.set(f"{self.imu_zero_pitch:+.2f}")
                        self.imu_status_yaw_base.set(f"{self.imu_zero_yaw:+.2f}")
                        self.imu_status_pitch_delta.set("+0.00")
                        self.imu_status_yaw_delta.set("+0.00")
                    elif key == "preview_jpg":
                        self.mp_preview_jpg = val
                        self.mp_preview_seq += 1
            except Exception:
                pass
            render_period = 1.0 / max(1.0, float(self.preview_render_hz.get()))
            now_ts = time.time()
            should_render = (
                not self.aggressive_perf_mode.get()
                and self.multiprocess_preview.get()
                and self.mp_preview_jpg is not None
                and self.mp_preview_seq != self._last_render_seq
                and (now_ts - self._last_render_ts) >= render_period
            )
            if should_render:
                try:
                    arr = np.frombuffer(self.mp_preview_jpg, dtype=np.uint8)
                    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img_bgr is not None:
                        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                        img_rgb = self._fit_preview_image(img_rgb)
                        image = Image.fromarray(img_rgb)
                        photo = ImageTk.PhotoImage(image=image)
                        self.preview_label.configure(image=photo, text="")
                        self.preview_label.image = photo
                        self._last_render_seq = int(self.mp_preview_seq)
                        self._last_render_ts = float(now_ts)
                except Exception:
                    pass
            elif self.aggressive_perf_mode.get():
                self.preview_label.configure(image="", text="AGGRESSIVE MODE")
                self.preview_label.image = None
            elif not self.multiprocess_preview.get():
                self.preview_label.configure(image="", text="MULTIPROCESS PREVIEW OFF")
                self.preview_label.image = None
            if self.mp_latest_detection is not None and self.mp_latest_detection[5] > 0.5:
                self.mp_det_age = max(0.0, time.time() - float(self.mp_latest_detection[4]))
            self.servo_status_mode.set("无刷RS485(MP)")
            self.servo_status_pan.set(f"{self.mp_pan:.1f}")
            self.servo_status_tilt.set(f"{self.mp_tilt:.1f}")
            self.servo_status_voltage.set("-")
            tracking_flag = int(bool(self.tracking_active) and bool(self.track_enabled.get()))
            try:
                period_ms = int(self.control_period_ms.get())
            except Exception:
                period_ms = 0
            self.status_text.set(
                f"多进程运行中 跟踪={tracking_flag} V={int(vision_alive)} C={int(control_alive)} 周期ms={period_ms} 视觉Hz={self.mp_vision_hz:.1f} 控制Hz={self.mp_control_hz:.1f} DetOK={int(self.mp_det_valid>0.5)} Age={self.mp_det_age:.3f}s UpdAge={self.mp_det_update_age:.3f}s 水平={self.mp_pan:.2f} 俯仰={self.mp_tilt:.2f} {self.mp_ctrl_timing}"
            )
            if self.mp_last_error:
                self.status_text.set(self.status_text.get() + f" 错误={self.mp_last_error}")
            self.after_id = self.root.after(self._ui_refresh_delay_ms(), self._ui_loop)
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
            render_period = 1.0 / max(1.0, float(self.preview_render_hz.get()))
            now_ts = time.time()
            render_now = (now_ts - float(getattr(self, "_last_render_ts", 0.0))) >= render_period
            if not self.aggressive_perf_mode.get():
                if render_now:
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
                        box = getattr(self, "_preview_box_size", None)
                        if box is None:
                            target_w, target_h = w, h
                        else:
                            target_w, target_h = int(box[0]), int(box[1])
                        target_w = max(160, target_w)
                        target_h = max(120, target_h)
                        panel_w = max(80, target_w // 2)
                        panel_h = max(60, target_h // 2)

                        main_panel = self._resize_crop(frame_rgb_disp, panel_w, panel_h)

                        green_gray = self._resize_crop(blurred_green, panel_w, panel_h)
                        red_gray = self._resize_crop(blurred_red, panel_w, panel_h)
                        green_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                        red_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)
                        green_panel[:, :, 1] = green_gray
                        red_panel[:, :, 0] = red_gray

                        if self.laser_align_mode.get():
                            _, binary = cv2.threshold(blurred_red, laser_threshold, 255, cv2.THRESH_BINARY)
                            bin_gray = self._resize_crop(binary, panel_w, panel_h, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST)
                            bin_panel = cv2.cvtColor(bin_gray, cv2.COLOR_GRAY2RGB)
                        else:
                            bin_panel = np.zeros((panel_h, panel_w, 3), dtype=np.uint8)

                        cv2.putText(main_panel, "Main", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
                        cv2.putText(green_panel, "Green", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(red_panel, "Red", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                        cv2.putText(bin_panel, "Laser Bin" if self.laser_align_mode.get() else "Bin Off", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)

                        top_row = np.hstack((main_panel, green_panel))
                        bottom_row = np.hstack((red_panel, bin_panel))
                        frame_rgb_show = np.vstack((top_row, bottom_row))
                    else:
                        frame_rgb_show = frame_rgb_disp

                    frame_rgb_show = self._fit_preview_image(frame_rgb_show)
                    image = Image.fromarray(frame_rgb_show)
                    photo = ImageTk.PhotoImage(image=image)
                    self.preview_label.configure(image=photo, text="")
                    self.preview_label.image = photo
                    self._last_render_ts = float(now_ts)
            else:
                self.preview_label.configure(image="", text="AGGRESSIVE MODE")
                self.preview_label.image = None
            
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

        self.after_id = self.root.after(self._ui_refresh_delay_ms(), self._ui_loop)

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
        bit_depth = int(settings.get("sensor_bit_depth", 10))
        target_size = getattr(self, "camera_target_size", (640, 640))
        raw_w = max(160, int(target_size[0]))
        raw_h = max(120, int(target_size[1]))
        frame_duration = int(1000000 / framerate)
        config = self.picam2.create_video_configuration(
            sensor={"output_size": (raw_w, raw_h), "bit_depth": int(bit_depth)},
            controls={"FrameDurationLimits": (frame_duration, frame_duration)}
        )
        config["main"]["format"] = "BGR888"
        config["main"]["size"] = (raw_w, raw_h)
        self.picam2.align_configuration(config)
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.2)
        self.last_ae_enable = None
        self.last_exposure = None
        self.last_gain = None
        self.last_fps = framerate
        self.camera_raw_size = (raw_w, raw_h)

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
        if self.multiprocess_mode.get():
            if self.mp_control_cmd_queue is not None:
                try:
                    self.mp_control_cmd_queue.put_nowait(("zero_imu_request",))
                    self.status_text.set("多进程IMU置零请求已发送")
                except Exception as exc:
                    self.worker_error = str(exc)
                    self.status_text.set(f"IMU置零请求失败: {exc}")
            return
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

    def _on_aggressive_perf_mode_toggle(self):
        if self.aggressive_perf_mode.get():
            self.show_debug_panels.set(False)
            self.preview_label.configure(image="", text="AGGRESSIVE MODE")
            self.preview_label.image = None
            self.status_text.set("已开启更加激进模式：关闭图像显示")
        else:
            self.preview_label.configure(text="")
            self.status_text.set("已关闭更加激进模式")

    def _apply_camera_resolution(self):
        try:
            w, h, adjusted = self._normalize_camera_resolution(self.camera_raw_width.get(), self.camera_raw_height.get())
            self.camera_raw_width.set(w)
            self.camera_raw_height.set(h)
            self.camera_target_size = (w, h)
            self.camera_reconfigure_event.set()
            if adjusted:
                self.status_text.set(f"分辨率已校正为16倍数: {w}x{h}，等待应用")
            else:
                self.status_text.set(f"相机分辨率待应用: {w}x{h}")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"分辨率配置失败: {exc}")

    def _normalize_camera_resolution(self, width, height):
        w = max(160, int(width))
        h = max(128, int(height))
        w_aligned = max(160, (w // 16) * 16)
        h_aligned = max(128, (h // 16) * 16)
        return w_aligned, h_aligned, (w_aligned != w or h_aligned != h)

    def _apply_imu_output_rate(self):
        if self.multiprocess_mode.get():
            self._push_multiprocess_settings()
            self.status_text.set(f"多进程已应用IMU输出速率: {int(self.imu_output_hz.get())}Hz")
            return
        try:
            self._ensure_imu()
            hz = int(self.imu_output_hz.get())
            self.imu.set_output_rate_hz(hz)
            self.status_text.set(f"IMU输出速率已设置为 {hz}Hz")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU配置失败: {exc}")

    def _apply_imu_algorithm_mode(self):
        if self.multiprocess_mode.get():
            self._push_multiprocess_settings()
            self.status_text.set("多进程已应用IMU算法模式")
            return
        try:
            self._ensure_imu()
            use_6axis = bool(self.imu_use_6axis.get())
            self.imu.set_algorithm_mode(use_6axis)
            self.status_text.set("IMU算法已设置为6轴" if use_6axis else "IMU算法已设置为9轴")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU配置失败: {exc}")

    def _apply_imu_baudrate(self):
        if self.multiprocess_mode.get():
            self._push_multiprocess_settings()
            self.status_text.set(f"多进程已应用IMU波特率: {int(self.imu_baudrate.get())}")
            return
        try:
            self._ensure_imu()
            baud = int(self.imu_baudrate.get())
            self.imu.apply_baudrate(baud)
            self.status_text.set(f"IMU波特率已设置为 {baud}")
        except Exception as exc:
            self.worker_error = str(exc)
            self.status_text.set(f"IMU配置失败: {exc}")

    def _apply_imu_offsets(self):
        if self.multiprocess_mode.get():
            self._push_multiprocess_settings()
            self.status_text.set("多进程已应用IMU零偏参数")
            return
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
        if self.multiprocess_mode.get():
            self.status_text.set("多进程模式下请先停止多进程再执行静置采样")
            return
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
        if self.servo_mode.get() != "无刷RS485":
            self.servo_mode.set("无刷RS485")
            return
        self._release_servo()

    def _refresh_servo_mode_ui(self):
        return

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
            self.hw_pan_min.set(-360.0)
            self.hw_pan_max.set(360.0)
            self.hw_tilt_min.set(-360.0)
            self.hw_tilt_max.set(360.0)
            self.status_text.set("无刷RS485已连接，物理边界固定±360°")
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
                moving_time=40,
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
            moving_time=40,
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

    def _fit_preview_image(self, img_rgb):
        box = getattr(self, "_preview_box_size", None)
        if box is None:
            try:
                box_w = int(self.preview_frame.winfo_width())
                box_h = int(self.preview_frame.winfo_height())
            except Exception:
                return img_rgb
        else:
            box_w, box_h = int(box[0]), int(box[1])
        if box_w < 64 or box_h < 64:
            return img_rgb
        return self._resize_crop(img_rgb, box_w, box_h)

    def _resize_crop(self, img, out_w, out_h, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_LINEAR):
        ih, iw = img.shape[:2]
        if iw <= 0 or ih <= 0:
            return img
        if iw == out_w and ih == out_h:
            return img
        scale = max(out_w / float(iw), out_h / float(ih))
        new_w = max(1, int(round(iw * scale)))
        new_h = max(1, int(round(ih * scale)))
        interp = interpolation_down if scale < 1.0 else interpolation_up
        resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
        x0 = max(0, (new_w - out_w) // 2)
        y0 = max(0, (new_h - out_h) // 2)
        return resized[y0 : y0 + out_h, x0 : x0 + out_w]

    def on_close(self):
        if self._is_closing:
            return
        self._is_closing = True
        self.tracking_active = False
        self._save_settings()
        if self.multiprocess_mode.get() and self.mp_control_cmd_queue is not None:
            try:
                self.mp_control_cmd_queue.put_nowait(("shutdown_pose",))
                time.sleep(0.35)
            except Exception:
                pass
        self.stop_event.set()
        self.detect_stop_event.set()
        self.stab_stop_event.set()
        self._stop_multiprocess_runtime()
        
        # 退出 GUI 前先走到关机姿态
        if self.servo is not None:
            try:
                print("[INFO] Exiting GUI. Moving to shutdown pose before motor off...")
                self._move_to_shutdown_pose()
            except Exception:
                pass
                
        if self.after_id is not None:
            try:
                self.root.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None
        if self.stab_thread is not None:
            self.stab_thread.join(timeout=2.0)
            self.stab_thread = None
        self._close_imu()
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
        self.root.destroy()

    def _jog(self, delta_pan, delta_tilt):
        if self.multiprocess_mode.get():
            if self.mp_control_cmd_queue is not None:
                try:
                    self.mp_control_cmd_queue.put_nowait(("jog", float(delta_pan), float(delta_tilt)))
                    self.status_text.set(f"多进程点动: pan={float(delta_pan):+.2f} tilt={float(delta_tilt):+.2f}")
                except Exception as exc:
                    self.worker_error = str(exc)
                    self.status_text.set(f"点动命令失败: {exc}")
            return
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
    def _handle_term(_signum, _frame):
        try:
            root.after(0, app.on_close)
        except Exception:
            try:
                app.on_close()
            except Exception:
                pass
    try:
        signal.signal(signal.SIGINT, _handle_term)
    except Exception:
        pass
    try:
        signal.signal(signal.SIGTERM, _handle_term)
    except Exception:
        pass
    try:
        root.mainloop()
    except KeyboardInterrupt:
        app.on_close()


if __name__ == "__main__":
    main()
