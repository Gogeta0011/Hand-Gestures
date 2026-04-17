# 🖐 Hand Gesture Control System

A real-time hand gesture recognition system built with Python, OpenCV, and MediaPipe. Control your mouse and media playback — no keyboard, no mouse, no contact.

---

## 🎮 Gesture Reference

| Gesture | Action |
| :--- | :--- |
| ☝️ **Index finger up** | Move mouse cursor |
| 👌 **Thumb + Index pinch** | Left click |
| 🤏 **Thumb + Middle pinch** | Right click |
| 👍 **Thumbs up** | Play / Pause |
| ✌️ **Peace (index + middle)** | Next track |
| 🤟 **Three fingers up** | Previous track |
| 🖐 **Open palm** | Mute / Unmute |
| ✊ **Fist** | Safety lock (disables all input) |
| 👉 **Swipe right** | Next track |
| 👈 **Swipe left** | Previous track |

---

## 🛠 Tech Stack

| Library | Purpose |
| :--- | :--- |
| **OpenCV** | Camera capture and visual overlay |
| **MediaPipe** | Real-time hand landmark detection |
| **PyAutoGUI** | Mouse movement and keyboard event injection |
| **NumPy** | Numerical utilities |

---

## ⚙️ Requirements

- Python 3.8+
- A webcam
- Windows (media keys via PyAutoGUI work best on Windows; some keys may differ on macOS/Linux)

---

## 🚀 Setup

**1. Clone the repository**
```bash
git clone https://github.com/Gogeta0011/hand-gesture-control.git
cd hand-gesture-control
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run**
```bash
python gesture_control.py
```

To stop at any time: press **ESC**, or move your mouse to the **top-left corner** of the screen (PyAutoGUI failsafe).

---

## 🔧 Configuration

All tunable parameters live at the top of `gesture_control.py`:

```python
CURSOR_SMOOTHING   = 0.25   # Higher = smoother but slower cursor response
COOLDOWN_CLICK     = 0.35   # Seconds between click events
COOLDOWN_MEDIA     = 0.70   # Seconds between media key presses
PINCH_LCLICK_THRESH = 0.045 # Pinch distance threshold for left click
PINCH_RCLICK_THRESH = 0.050 # Pinch distance threshold for right click
SWIPE_MIN_DX       = 0.12   # Minimum normalized wrist movement for swipe
STOP_ON_FIST       = True   # Enable/disable fist safety lock
```

---

## 🏗 Architecture

```
Webcam Frame
     │
     ▼
MediaPipe Hands  ──►  21 Landmark Coordinates
     │
     ▼
Gesture Classifier
     │
     ├──► INDEX     →  PyAutoGUI mouse move  (smoothed)
     ├──► PINCH     →  PyAutoGUI click
     ├──► THUMBS_UP →  Play / Pause key
     ├──► PEACE     →  Next track key
     ├──► THREE     →  Prev track key
     ├──► PALM      →  Mute key
     ├──► SWIPE     →  Next / Prev track key
     └──► FIST      →  Safety lock (no output)
```

---

## 💡 Tips

- Use in a well-lit environment for best tracking accuracy.
- Keep your hand within ~60cm of the webcam.
- The fist gesture acts as a deadman switch — make a fist to instantly pause all gesture input.
- Adjust `CURSOR_SMOOTHING` if the cursor feels laggy or jittery.
