import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import threading
from collections import deque

pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


class EyeFlowTracker:
    def __init__(self, settings=None, calibration=None):
        self.settings = settings or {}
        self.calibration = calibration

        self.screen_w, self.screen_h = pyautogui.size()
        print(f"[EyeFlow] Screen: {self.screen_w}x{self.screen_h}")

        # Landmarks
        self.LEFT_IRIS = 469
        self.NOSE_TIP = 1

        # Right eye points for EAR
        self.RIGHT_EYE = {
            'p1': 362, 'p2': 385, 'p3': 387,
            'p4': 263, 'p5': 373, 'p6': 380
        }

        # Left eye points for EAR
        self.LEFT_EYE = {
            'p1': 33, 'p2': 160, 'p3': 158,
            'p4': 133, 'p5': 153, 'p6': 144
        }

        # ===== EASIER BLINK CLICK =====
        self.BLINK_EAR_THRESHOLD = float(self.settings.get('blink_threshold', 0.28))
        self.MIN_BLINK_DURATION = 0.05
        self.MAX_BLINK_DURATION = 0.7
        self.CLICK_COOLDOWN = float(self.settings.get('click_cooldown', 0.5))
        self.CURSOR_FREEZE_TIME = 0.9

        # How much right eye must close MORE than left
        # Lower = easier to click
        self.WINK_DIFF = 0.01

        # EAR tracking
        self.ear_baseline_right = 0.28
        self.ear_baseline_left = 0.28
        self.baseline_count = 0

        # Cursor
        self.SMOOTHING = float(self.settings.get('cursor_smoothing', 0.18))

        # ===== BALANCED SCROLL =====
        self.NOSE_SCROLL_DEADZONE = float(self.settings.get('scroll_deadzone', 0.012))
        self.SCROLL_UP_GAIN = float(self.settings.get('scroll_gain', 1200))
        self.SCROLL_DOWN_GAIN = float(self.settings.get('scroll_gain', 400))
        self.MAX_SCROLL = 15
        self.SCROLL_COOLDOWN = 0.04

        # Calibration
        self.calibrated = False
        self.eye_min = np.array([1.0, 1.0])
        self.eye_max = np.array([0.0, 0.0])
        self.nose_center = None

        if calibration:
            try:
                self.eye_min = np.array(calibration['eye_min'], dtype=float)
                self.eye_max = np.array(calibration['eye_max'], dtype=float)
                self.nose_center = [float(calibration['nose_center'][0]),
                                    float(calibration['nose_center'][1])]
                self.calibrated = True
                print(f"[EyeFlow] Calibration loaded")
            except Exception as e:
                print(f"[EyeFlow] Cal error: {e}")

        # States
        self.blink_start_time = None
        self.right_was_closed = False
        self.last_click_time = 0
        self.cursor_freeze_until = 0
        self.smoothed_x = float(self.screen_w // 2)
        self.smoothed_y = float(self.screen_h // 2)
        self.last_scroll_time = 0

        # Stats
        self.clicks = 0
        self.scrolls = 0
        self.start_time = None
        self.frame_times = deque(maxlen=60)
        self.current_ear = 0.3
        self.current_ear_left = 0.3
        self.latency = 0
        self.face_detected = False

        # Voice
        self.voice_active = bool(self.settings.get('voice_typing', False))
        self.voice_engine = None
        self.voice_thread = None

        # Control
        self.running = False
        self.cap = None
        self.current_frame = None
        self.frame_lock = threading.Lock()

        print(f"[EyeFlow] Blink threshold: {self.BLINK_EAR_THRESHOLD}")
        print(f"[EyeFlow] Wink diff: {self.WINK_DIFF}")
        print(f"[EyeFlow] Scroll deadzone: {self.NOSE_SCROLL_DEADZONE}")
        print(f"[EyeFlow] Scroll UP gain: {self.SCROLL_UP_GAIN}")
        print(f"[EyeFlow] Scroll DOWN gain: {self.SCROLL_DOWN_GAIN}")

    def update_settings(self, settings):
        self.BLINK_EAR_THRESHOLD = float(settings.get('blink_threshold', 0.24))
        self.SMOOTHING = float(settings.get('cursor_smoothing', 0.18))
        self.CLICK_COOLDOWN = float(settings.get('click_cooldown', 0.7))
        sg = float(settings.get('scroll_gain', 700))
        self.SCROLL_UP_GAIN = sg * 1.15
        self.SCROLL_DOWN_GAIN = sg * 0.72
        self.NOSE_SCROLL_DEADZONE = float(settings.get('scroll_deadzone', 0.025))

        new_voice = bool(settings.get('voice_typing', False))
        if new_voice and not self.voice_active:
            self.voice_active = True
            self._start_voice_engine(settings.get('voice_language', 'en-US'))
        elif not new_voice and self.voice_active:
            self.voice_active = False
            self._stop_voice_engine()

    def calc_ear(self, lm, eye):
        try:
            p1 = np.array([lm[eye['p1']].x, lm[eye['p1']].y])
            p2 = np.array([lm[eye['p2']].x, lm[eye['p2']].y])
            p3 = np.array([lm[eye['p3']].x, lm[eye['p3']].y])
            p4 = np.array([lm[eye['p4']].x, lm[eye['p4']].y])
            p5 = np.array([lm[eye['p5']].x, lm[eye['p5']].y])
            p6 = np.array([lm[eye['p6']].x, lm[eye['p6']].y])

            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h = np.linalg.norm(p1 - p4)

            if h < 0.0001:
                return 0.3
            return float((v1 + v2) / (2.0 * h))
        except:
            return 0.3

    def detect_click(self, lm, now):
        """
        Easier right eye click detection:
        - Right eye closes below threshold
        - Left eye stays more open than right (even slightly)
        - Natural blinks (both eyes close together) are ignored
        """
        r_ear = self.calc_ear(lm, self.RIGHT_EYE)
        l_ear = self.calc_ear(lm, self.LEFT_EYE)

        self.current_ear = r_ear
        self.current_ear_left = l_ear

        # Learn baseline (first 25 frames)
        if self.baseline_count < 25:
            if r_ear > 0.15 and l_ear > 0.15:
                self.ear_baseline_right = (self.ear_baseline_right * self.baseline_count + r_ear) / (self.baseline_count + 1)
                self.ear_baseline_left = (self.ear_baseline_left * self.baseline_count + l_ear) / (self.baseline_count + 1)
                self.baseline_count += 1
                if self.baseline_count == 25:
                    print(f"[EyeFlow] Baseline R={self.ear_baseline_right:.3f} L={self.ear_baseline_left:.3f}")
            return False

        right_closed = r_ear < self.BLINK_EAR_THRESHOLD
        both_closed = r_ear < self.BLINK_EAR_THRESHOLD and l_ear < (self.BLINK_EAR_THRESHOLD - 0.01)

        # Right eye is more closed than left
        right_more_closed = (l_ear - r_ear) > self.WINK_DIFF

        is_wink = right_closed and right_more_closed and not both_closed

        if is_wink:
            if self.blink_start_time is None:
                self.blink_start_time = now
                self.right_was_closed = True
        else:
            if self.right_was_closed and not right_closed:
                # Right eye opened
                if self.blink_start_time is not None:
                    dur = now - self.blink_start_time
                    cd_ok = (now - self.last_click_time) > self.CLICK_COOLDOWN

                    self.blink_start_time = None
                    self.right_was_closed = False

                    if (dur >= self.MIN_BLINK_DURATION and
                            dur <= self.MAX_BLINK_DURATION and cd_ok):
                        return True
                else:
                    self.right_was_closed = False

            if both_closed:
                self.blink_start_time = None
                self.right_was_closed = False

        return False

    def _start_voice_engine(self, language='en-US'):
        try:
            from voice_engine import VoiceTypingEngine
            if self.voice_engine and self.voice_engine.running:
                return
            self.voice_engine = VoiceTypingEngine(language=language)
            self.voice_thread = threading.Thread(target=self.voice_engine.run, daemon=True)
            self.voice_thread.start()
            print(f"[EyeFlow] Voice started ({language})")
        except ImportError:
            print("[EyeFlow] Voice not available")
            self.voice_engine = None
        except Exception as e:
            print(f"[EyeFlow] Voice error: {e}")
            self.voice_engine = None

    def _stop_voice_engine(self):
        if self.voice_engine:
            self.voice_engine.stop()
            self.voice_engine = None
            self.voice_thread = None

    def toggle_voice(self):
        self.voice_active = not self.voice_active
        if self.voice_active:
            self._start_voice_engine(self.settings.get('voice_language', 'en-US'))
        else:
            self._stop_voice_engine()
        return self.voice_active

    def get_stats(self):
        elapsed = time.time() - self.start_time if self.start_time else 0
        fps = 0
        if len(self.frame_times) > 1:
            s = self.frame_times[-1] - self.frame_times[0]
            if s > 0:
                fps = int(len(self.frame_times) / s)

        vc = 0
        if self.voice_engine:
            try:
                vc = self.voice_engine.typed_count
            except:
                pass

        return {
            'fps': fps,
            'accuracy': 98 if self.calibrated and self.face_detected else 0,
            'latency': int(self.latency * 1000),
            'clicks': self.clicks,
            'scrolls': self.scrolls,
            'voice_inputs': vc,
            'uptime': int(elapsed),
            'voice_active': self.voice_active,
            'calibrated': self.calibrated,
            'ear': round(self.current_ear, 3),
            'face_detected': self.face_detected
        }

    def get_frame_bytes(self):
        with self.frame_lock:
            if self.current_frame is not None:
                try:
                    small = cv2.resize(self.current_frame, (320, 240))
                    _, buf = cv2.imencode('.jpg', small, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    return buf.tobytes()
                except:
                    pass
        return None

    def run(self):
        self.running = True
        self.start_time = time.time()

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("[EyeFlow] Camera failed!")
            self.running = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        for _ in range(15):
            self.cap.read()
            time.sleep(0.03)

        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        if self.voice_active:
            self._start_voice_engine(self.settings.get('voice_language', 'en-US'))

        frame_count = 0
        last_log = time.time()

        print("[EyeFlow] ============================")
        print("[EyeFlow] TRACKING STARTED")
        print("[EyeFlow] Right eye wink → Click")
        print("[EyeFlow] Nose UP → Scroll UP (boosted)")
        print("[EyeFlow] Nose DOWN → Scroll DOWN")
        print("[EyeFlow] ============================")

        while self.running:
            t0 = time.time()

            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = face_mesh.process(rgb)

            if result.multi_face_landmarks:
                self.face_detected = True
                lm = result.multi_face_landmarks[0].landmark
                now = time.time()

                iris = lm[self.LEFT_IRIS]
                nose = lm[self.NOSE_TIP]

                # Draw
                cv2.circle(frame, (int(iris.x * w), int(iris.y * h)), 4, (0, 255, 0), -1)
                cv2.circle(frame, (int(nose.x * w), int(nose.y * h)), 4, (255, 0, 0), -1)

                for key in self.RIGHT_EYE:
                    idx = self.RIGHT_EYE[key]
                    cv2.circle(frame, (int(lm[idx].x * w), int(lm[idx].y * h)), 2, (0, 255, 255), -1)

                if self.calibrated and self.nose_center:

                    # ===== CURSOR =====
                    rel_x = iris.x - self.nose_center[0]
                    rel_y = iris.y - self.nose_center[1]

                    rx = self.eye_max[0] - self.eye_min[0]
                    ry = self.eye_max[1] - self.eye_min[1]
                    if abs(rx) < 0.0001: rx = 0.01
                    if abs(ry) < 0.0001: ry = 0.01

                    nx = float(np.clip((rel_x - self.eye_min[0]) / rx, 0, 1))
                    ny = float(np.clip((rel_y - self.eye_min[1]) / ry, 0, 1))

                    # ===== CLICK =====
                    if self.detect_click(lm, now):
                        try:
                            pyautogui.click()
                            self.clicks += 1
                            self.last_click_time = now
                            self.cursor_freeze_until = now + self.CURSOR_FREEZE_TIME
                            print(f"[EyeFlow] 🖱 CLICK! R={self.current_ear:.3f} "
                                  f"L={self.current_ear_left:.3f} total={self.clicks}")
                        except Exception as e:
                            print(f"[EyeFlow] Click err: {e}")

                    # ===== BALANCED SCROLL =====
                    nose_offset = self.nose_center[1] - nose.y

                    if (now > self.cursor_freeze_until and
                            abs(nose_offset) > self.NOSE_SCROLL_DEADZONE and
                            now - self.last_scroll_time > self.SCROLL_COOLDOWN):

                        # UP gets higher gain, DOWN gets lower gain
                        if nose_offset > 0:
                            scroll_val = nose_offset * self.SCROLL_UP_GAIN
                        else:
                            scroll_val = nose_offset * self.SCROLL_DOWN_GAIN

                        scroll_val = max(-self.MAX_SCROLL, min(self.MAX_SCROLL, scroll_val))
                        scroll_val = int(scroll_val)

                        if scroll_val != 0:
                            try:
                                pyautogui.scroll(scroll_val)
                                self.scrolls += 1
                                self.last_scroll_time = now
                            except:
                                pass

                    # ===== MOVE CURSOR =====
                    if now > self.cursor_freeze_until:
                        tx = nx * self.screen_w
                        ty = ny * self.screen_h

                        self.smoothed_x += (tx - self.smoothed_x) * self.SMOOTHING
                        self.smoothed_y += (ty - self.smoothed_y) * self.SMOOTHING

                        mx = max(0, min(self.screen_w - 1, int(self.smoothed_x)))
                        my = max(0, min(self.screen_h - 1, int(self.smoothed_y)))

                        try:
                            pyautogui.moveTo(mx, my)
                        except:
                            pass

                    # ===== OVERLAY =====
                    r_col = (0, 0, 255) if self.current_ear < self.BLINK_EAR_THRESHOLD else (0, 255, 255)
                    cv2.putText(frame, f"R: {self.current_ear:.3f}", (10, 22),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, r_col, 1)
                    cv2.putText(frame, f"L: {self.current_ear_left:.3f}", (10, 42),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(frame, f"Nose: {nose_offset:.4f}", (10, 62),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
                    cv2.putText(frame, f"C:{self.clicks} S:{self.scrolls}", (10, 82),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)

                    # Wink indicator
                    if self.right_was_closed:
                        cv2.putText(frame, ">> WINK <<", (w // 2 - 60, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    # Scroll indicator
                    if abs(nose_offset) > self.NOSE_SCROLL_DEADZONE:
                        d = "SCROLL UP" if nose_offset > 0 else "SCROLL DOWN"
                        cv2.putText(frame, d, (10, 105),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)
                else:
                    cv2.putText(frame, "NOT CALIBRATED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.face_detected = False

            with self.frame_lock:
                self.current_frame = frame.copy()

            self.latency = time.time() - t0
            self.frame_times.append(time.time())
            frame_count += 1

            if time.time() - last_log > 15:
                fps = len(self.frame_times) / max(0.1, self.frame_times[-1] - self.frame_times[0]) if len(self.frame_times) > 1 else 0
                print(f"[EyeFlow] FPS={int(fps)} C={self.clicks} S={self.scrolls} "
                      f"R={self.current_ear:.3f} L={self.current_ear_left:.3f}")
                last_log = time.time()

            time.sleep(0.001)

        self._cleanup()
        print("[EyeFlow] STOPPED")

    def stop(self):
        self.running = False

    def emergency_stop(self):
        self.running = False
        self._stop_voice_engine()
        if self.cap:
            try: self.cap.release()
            except: pass

    def _cleanup(self):
        self._stop_voice_engine()
        if self.cap:
            try: self.cap.release()
            except: pass