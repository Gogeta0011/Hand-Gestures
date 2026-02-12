import time
import math
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = True  # move mouse to top-left to instantly stop
pyautogui.PAUSE = 0        # we handle timing ourselves


# ------------------------------ Config ------------------------------
CAM_INDEX = 0
FRAME_W, FRAME_H = 1280, 720

# Cursor smoothing (higher = smoother, slower response)
CURSOR_SMOOTHING = 0.25

# Gesture debounce/cooldowns (seconds)
COOLDOWN_CLICK = 0.35
COOLDOWN_MEDIA = 0.70
COOLDOWN_MUTE  = 0.90

# Swipe detection
SWIPE_WINDOW = 8                  # frames
SWIPE_MIN_DX = 0.12               # normalized (0..1) movement threshold
SWIPE_COOLDOWN = 0.9

# Pinch thresholds (normalized distance; smaller means fingers close)
PINCH_LCLICK_THRESH = 0.045
PINCH_RCLICK_THRESH = 0.050

# Finger "up" heuristic thresholds
FINGER_UP_Y_DELTA = 0.02          # how much fingertip must be above pip

# Safety
STOP_ON_FIST = True

# -------------------------------------------------------------------

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

def lm_xy(lm):
    return (lm.x, lm.y)

def finger_is_up(lms, tip_idx, pip_idx):
    # Tip above PIP (y smaller is higher in image coords)
    return (lms[tip_idx].y + FINGER_UP_Y_DELTA) < lms[pip_idx].y

def thumb_is_up(lms):
    # Use x-direction heuristic depending on which hand (we’ll infer handedness by thumb position)
    # If thumb tip is to the left of MCP, treat as "extended" for one orientation; otherwise opposite.
    tip = lms[4]
    mcp = lms[2]
    # Extended if sufficiently far in x
    return abs(tip.x - mcp.x) > 0.06

def classify_static_gesture(lms):
    """
    Returns a label among:
    - 'FIST', 'PALM', 'THUMBS_UP', 'PEACE', 'THREE', 'INDEX', 'OTHER'
    """
    idx_up = finger_is_up(lms, 8, 6)
    mid_up = finger_is_up(lms, 12, 10)
    ring_up = finger_is_up(lms, 16, 14)
    pinky_up = finger_is_up(lms, 20, 18)
    th_up = thumb_is_up(lms)

    up_count = sum([idx_up, mid_up, ring_up, pinky_up])

    # Fist: no fingers up (thumb may vary)
    if up_count == 0 and not idx_up and not mid_up and not ring_up and not pinky_up:
        return "FIST"

    # Palm: all four fingers up (thumb optional)
    if up_count == 4:
        return "PALM"

    # Index only: cursor move mode
    if idx_up and not mid_up and not ring_up and not pinky_up:
        return "INDEX"

    # Peace: index + middle
    if idx_up and mid_up and not ring_up and not pinky_up:
        # If thumb is strongly extended it might be "gun" pose; still treat as PEACE for simplicity
        return "PEACE"

    # Three fingers: index + middle + ring
    if idx_up and mid_up and ring_up and not pinky_up:
        return "THREE"

    # Thumbs up: thumb extended, other fingers down
    if th_up and up_count == 0:
        return "THUMBS_UP"

    return "OTHER"

class Cooldowns:
    def __init__(self):
        self.t = {}

    def ready(self, key, cooldown_s):
        now = time.time()
        last = self.t.get(key, 0.0)
        if now - last >= cooldown_s:
            self.t[key] = now
            return True
        return False

class SwipeDetector:
    def __init__(self, window=SWIPE_WINDOW):
        self.window = window
        self.x_hist = deque(maxlen=window)
        self.t_hist = deque(maxlen=window)
        self.last_swipe = 0.0

    def update(self, x_norm):
        self.x_hist.append(x_norm)
        self.t_hist.append(time.time())

    def detect(self):
        if len(self.x_hist) < self.window:
            return None

        now = time.time()
        if now - self.last_swipe < SWIPE_COOLDOWN:
            return None

        dx = self.x_hist[-1] - self.x_hist[0]
        if abs(dx) < SWIPE_MIN_DX:
            return None

        self.last_swipe = now
        return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"

def main():
    cap = cv2.VideoCapture(CAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

    screen_w, screen_h = pyautogui.size()

    cooldowns = Cooldowns()
    swipe = SwipeDetector()

    # Smoothed cursor position
    cur_x, cur_y = pyautogui.position()

    with mp_hands.Hands(
        model_complexity=1,
        min_detection_confidence=0.65,
        min_tracking_confidence=0.65,
        max_num_hands=1
    ) as hands:
        last_fps_t = time.time()
        fps = 0.0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)  # mirror for natural control
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            label = "NO_HAND"
            info_lines = []

            if res.multi_hand_landmarks:
                hand_lms = res.multi_hand_landmarks[0]
                lms = hand_lms.landmark

                mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

                label = classify_static_gesture(lms)

                # Key points (normalized)
                idx_tip = lm_xy(lms[8])
                th_tip  = lm_xy(lms[4])
                mid_tip = lm_xy(lms[12])

                # Pinch distances
                pinch_idx = dist(idx_tip, th_tip)
                pinch_mid = dist(mid_tip, th_tip)

                # Swipe tracking on wrist x (more stable)
                wrist_x = lms[0].x
                swipe.update(wrist_x)
                swipe_event = swipe.detect()

                # Safety lock
                if STOP_ON_FIST and label == "FIST":
                    info_lines.append("LOCKED (FIST)")
                else:
                    # ---------------- Mouse Move (INDEX mode) ----------------
                    if label == "INDEX":
                        # Map index tip to screen coordinates
                        x = clamp(idx_tip[0], 0.0, 1.0)
                        y = clamp(idx_tip[1], 0.0, 1.0)

                        target_x = int(x * screen_w)
                        target_y = int(y * screen_h)

                        # Smooth
                        cur_x = int(cur_x + (target_x - cur_x) * CURSOR_SMOOTHING)
                        cur_y = int(cur_y + (target_y - cur_y) * CURSOR_SMOOTHING)

                        pyautogui.moveTo(cur_x, cur_y)
                        info_lines.append("MOUSE MOVE")

                    # ---------------- Clicks (Pinch) ----------------
                    # Left click: thumb + index pinch
                    if pinch_idx < PINCH_LCLICK_THRESH and cooldowns.ready("lclick", COOLDOWN_CLICK):
                        pyautogui.click()
                        info_lines.append("LEFT CLICK")

                    # Right click: thumb + middle pinch
                    if pinch_mid < PINCH_RCLICK_THRESH and cooldowns.ready("rclick", COOLDOWN_CLICK):
                        pyautogui.click(button="right")
                        info_lines.append("RIGHT CLICK")

                    # ---------------- Media Controls ----------------
                    # Play/Pause
                    if label == "THUMBS_UP" and cooldowns.ready("playpause", COOLDOWN_MEDIA):
                        pyautogui.press("playpause")
                        info_lines.append("MEDIA: PLAY/PAUSE")

                    # Next / Previous
                    if label == "PEACE" and cooldowns.ready("next", COOLDOWN_MEDIA):
                        pyautogui.press("nexttrack")
                        info_lines.append("MEDIA: NEXT")

                    if label == "THREE" and cooldowns.ready("prev", COOLDOWN_MEDIA):
                        pyautogui.press("prevtrack")
                        info_lines.append("MEDIA: PREV")

                    # Mute on PALM (optional)
                    if label == "PALM" and cooldowns.ready("mute", COOLDOWN_MUTE):
                        pyautogui.press("volumemute")
                        info_lines.append("MEDIA: MUTE")

                    # Swipe to also do next/prev (nice demo)
                    if swipe_event == "SWIPE_RIGHT" and cooldowns.ready("swipe_next", COOLDOWN_MEDIA):
                        pyautogui.press("nexttrack")
                        info_lines.append("SWIPE: NEXT")

                    if swipe_event == "SWIPE_LEFT" and cooldowns.ready("swipe_prev", COOLDOWN_MEDIA):
                        pyautogui.press("prevtrack")
                        info_lines.append("SWIPE: PREV")

                # Debug overlay
                info_lines.append(f"Gesture: {label}")
                info_lines.append(f"Pinch idx: {pinch_idx:.3f}  mid: {pinch_mid:.3f}")

            # FPS
            now = time.time()
            dt = now - last_fps_t
            if dt > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / dt)
            last_fps_t = now

            # UI overlay
            cv2.rectangle(frame, (10, 10), (520, 140), (0, 0, 0), -1)
            cv2.putText(frame, f"FPS: {fps:.1f}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            y0 = 70
            for i, line in enumerate(info_lines[:4]):
                cv2.putText(frame, line, (20, y0 + i * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            cv2.putText(frame, "ESC to quit | Move mouse to top-left to FAILSAFE stop",
                        (10, frame.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (255, 255, 255), 2)

            cv2.imshow("Hand Gesture Control (Mouse + Media)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
