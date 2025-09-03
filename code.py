"""
smart_eye_drop_final.py

What this does:
- Talks continuously with clear guidance (threaded TTS queue; no blocking)
- Detects left/right eye, guides alignment, and determines open vs closed via EAR
- Verifies a drop by detecting motion entering the eye region from above, then logs it
- Debounces states + smooths signals to stop flicker and double counting
- Logs to drop_log.csv with timestamp, date, eye, success

Tips:
- If speech is too chatty, set TALK_EVERY_CHANGE=True -> False or increase TALK_COOLDOWN.
- If it says eyes are closed when open, LOWER EAR_OPEN_THRESH a bit (e.g., 0.22 -> 0.18).
- If it misses real drops, LOWER DROP_DIFF_THRESH slightly (e.g., 18 -> 12) and ensure good lighting.

Requirements (installed in your venv earlier):
  mediapipe opencv-python numpy pyttsx3

Press Q to quit the camera window.
"""

import cv2, mediapipe as mp, numpy as np, time, csv, threading, os
from datetime import datetime, date
from queue import Queue

# ===================== CONFIG (tune here) =====================
# Eye/Alignment
EAR_OPEN_THRESH      = 0.22   # lower if it says "closed" while open; raise if too permissive
EMA_ALPHA            = 0.35   # smoothing for EAR and centers (0<alpha<=1); higher = faster
ALIGN_PIX_THRESHOLD  = 36     # allowed pixels from frame-center for "Good alignment"
DEBOUNCE_FRAMES      = 6      # consecutive frames required to confirm READY
STABLE_HOLD_SECONDS  = 0.7    # hold steady this long before "dispensing"

# Speech
TALK_EVERY_CHANGE    = True   # speak only when guidance text changes
TALK_COOLDOWN        = 0.9    # seconds min between speech events to reduce chatter

# Drop detection (simple, demo-friendly)
DROP_DIFF_THRESH     = 18.0   # motion threshold (mean abs diff in eye ROI)
DROP_MIN_AREA        = 25     # ignore tiny motion blobs (pixels)
ENTRY_FROM_ABOVE_BIAS= 0.1    # fraction of ROI height defining "top band" to qualify as from above

# Cooldowns / logging
PER_EYE_COOLDOWN     = 4.0    # seconds before same eye can be counted again
LOGFILE              = "drop_log.csv"

# Camera backend (Windows often benefits from CAP_DSHOW)
CAPTURE_INDEX        = 0
CAPTURE_BACKEND      = cv2.CAP_DSHOW  # or 0 on non-Windows

# ===================== UTILITIES =====================
def ensure_log():
    if not os.path.exists(LOGFILE):
        with open(LOGFILE, "w", newline="") as f:
            csv.writer(f).writerow(["timestamp","date","eye","success"])

def log_event(eye, success):
    with open(LOGFILE, "a", newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), date.today().isoformat(), eye, int(success)])

class TTSQueue:
    """One pyttsx3 engine in a background thread; enqueue text to speak without blocking."""
    def __init__(self, cooldown=TALK_COOLDOWN):
        self.q = Queue()
        self.cooldown = cooldown
        self._last_spoke = 0.0
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            import pyttsx3
            engine = pyttsx3.init()
            # Optional: slow it slightly so words are clear (uncomment to tweak)
            # rate = engine.getProperty('rate')
            # engine.setProperty('rate', rate - 20)
        except Exception as e:
            print("pyttsx3 init failed; will print instead of speaking.", e)
            engine = None
        while True:
            item = self.q.get()
            if item is None:
                break
            text = item
            if engine:
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as e:
                    print("TTS error:", e)
            else:
                print("[SPEAK]", text)

    def say(self, text):
        now = time.time()
        if now - self._last_spoke >= self.cooldown:
            self.q.put(text)
            self._last_spoke = now

    def stop(self):
        self.q.put(None)

class EMA:
    """EMA that supports scalars or 2D tuples."""
    def __init__(self, alpha=0.35):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x):
        if x is None: return self.v
        if isinstance(x, (tuple, list, np.ndarray)):
            arr = np.array(x, dtype=float)
            if self.v is None:
                self.v = arr
            else:
                self.v = self.alpha * arr + (1 - self.alpha) * self.v
            return tuple(self.v.tolist())
        else:
            xv = float(x)
            if self.v is None:
                self.v = xv
            else:
                self.v = self.alpha * xv + (1 - self.alpha) * self.v
            return float(self.v)
    def value(self):
        if self.v is None: return None
        if isinstance(self.v, np.ndarray):
            return tuple(self.v.tolist())
        return float(self.v)

def mean_abs_diff(gray1, gray2):
    if gray1 is None or gray2 is None: return 0.0
    if gray1.shape != gray2.shape:
        gray2 = cv2.resize(gray2, (gray1.shape[1], gray1.shape[0]))
    return float(cv2.mean(cv2.absdiff(gray1, gray2))[0])

def eye_center(points):
    if not points: return None
    xs = [p[0] for p in points]; ys = [p[1] for p in points]
    return (int(sum(xs)/len(xs)), int(sum(ys)/len(ys)))

def euclid(a, b):
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))

# ===================== MEDIAPIPE & EAR =====================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
)

# EAR landmarks (MediaPipe indices)
# Left: 33(p1), 160(p2), 158(p3), 133(p4), 153(p5), 144(p6)
# Right: 362(p1), 385(p2), 387(p3), 263(p4), 373(p5), 380(p6)
LEFT_EAR = [33, 160, 158, 133, 153, 144]
RIGHT_EAR= [362,385, 387, 263, 373, 380]

def get_eye_points(landmarks, idxs, W, H):
    pts = []
    for i in idxs:
        if i < len(landmarks):
            p = landmarks[i]
            pts.append((int(p.x*W), int(p.y*H)))
    return pts if len(pts) == len(idxs) else None

def compute_EAR(pts):
    # Using (p1,p2,p3,p4,p5,p6) order
    p1,p2,p3,p4,p5,p6 = pts
    # EAR formula: ( ||p2-p6|| + ||p3-p5|| ) / ( 2 * ||p1-p4|| )
    numerator = euclid(p2,p6) + euclid(p3,p5)
    denominator = 2.0 * euclid(p1,p4) + 1e-6
    return numerator / denominator

# ===================== STATE =====================
STATE_ALIGNING   = "ALIGNING"
STATE_READY      = "READY"
STATE_DISPENSING = "DISPENSING"
STATE_VERIFYING  = "VERIFYING"
STATE_COOLDOWN   = "COOLDOWN"

class EyeState:
    def __init__(self):
        self.ear_ema = EMA(EMA_ALPHA)
        self.center_ema = EMA(EMA_ALPHA)
        self.debounce = 0
        self.last_dispense = 0.0

left_eye_state  = EyeState()
right_eye_state = EyeState()

# ===================== MAIN =====================
ensure_log()
tts = TTSQueue()

cap = cv2.VideoCapture(CAPTURE_INDEX, CAPTURE_BACKEND)
if not cap.isOpened():
    # fallback
    cap = cv2.VideoCapture(CAPTURE_INDEX)

tts.say("Smart dispenser started. Please face the camera.")

global_state = STATE_ALIGNING
state_since  = time.time()
last_guidance_text = None

prev_roi_gray = None  # for motion diff

def speak_guidance(text):
    global last_guidance_text
    if TALK_EVERY_CHANGE:
        if text != last_guidance_text:
            tts.say(text)
            last_guidance_text = text
    else:
        tts.say(text)

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        H, W = frame.shape[:2]
        cx, cy = W//2, H//2
        frame_center = (cx, cy)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        guidance = "Align your face"
        target_eye = "left"  # default
        target_state = left_eye_state
        left_pts = right_pts = None

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            left_pts6  = get_eye_points(lms, LEFT_EAR,  W, H)
            right_pts6 = get_eye_points(lms, RIGHT_EAR, W, H)

            # Draw eye outlines (convex hull for clarity)
            if left_pts6:
                hull = cv2.convexHull(np.array(left_pts6, dtype=np.int32))
                cv2.polylines(frame, [hull], True, (0,255,0), 1)
                # also get a denser polygon for ROI (use hull)
                left_pts = hull.reshape(-1,2).tolist()
            if right_pts6:
                hull = cv2.convexHull(np.array(right_pts6, dtype=np.int32))
                cv2.polylines(frame, [hull], True, (0,255,0), 1)
                right_pts = hull.reshape(-1,2).tolist()

            # Compute EAR & center for each eye
            if left_pts6:
                left_center_raw = eye_center(left_pts6)
                left_ear_raw    = compute_EAR(left_pts6)
                left_eye_state.center_ema.update(left_center_raw)
                left_eye_state.ear_ema.update(left_ear_raw)
            if right_pts6:
                right_center_raw = eye_center(right_pts6)
                right_ear_raw    = compute_EAR(right_pts6)
                right_eye_state.center_ema.update(right_center_raw)
                right_eye_state.ear_ema.update(right_ear_raw)

            # Pick target eye by which center is nearer to frame center (simple demo heuristic)
            lcen = left_eye_state.center_ema.value()
            rcen = right_eye_state.center_ema.value()
            ldist = np.inf if lcen is None else np.hypot(lcen[0]-cx, lcen[1]-cy)
            rdist = np.inf if rcen is None else np.hypot(rcen[0]-cx, rcen[1]-cy)

            if ldist <= rdist:
                target_eye   = "left"
                target_state = left_eye_state
                eye_pts_hull = left_pts if left_pts else left_pts6
            else:
                target_eye   = "right"
                target_state = right_eye_state
                eye_pts_hull = right_pts if right_pts else right_pts6

            # Alignment guidance
            cen = target_state.center_ema.value()
            if cen is not None:
                dx = cen[0] - cx
                dy = cen[1] - cy
                if abs(dx) > ALIGN_PIX_THRESHOLD:
                    guidance = "Move slightly " + ("left" if dx > 0 else "right")
                elif abs(dy) > ALIGN_PIX_THRESHOLD:
                    guidance = "Move slightly " + ("up" if dy > 0 else "down")
                else:
                    guidance = "Good alignment"
            else:
                guidance = "Align your face"

            # Open/closed via EAR
            ear = target_state.ear_ema.value() or 0.0
            eye_open = ear >= EAR_OPEN_THRESH

            # Debounce READY condition
            ready_now = (guidance == "Good alignment") and eye_open
            if ready_now:
                target_state.debounce += 1
            else:
                target_state.debounce = 0

            # State transitions
            now = time.time()
            if target_state.debounce >= DEBOUNCE_FRAMES and (now - target_state.last_dispense) > PER_EYE_COOLDOWN:
                if global_state != STATE_READY:
                    global_state = STATE_READY
                    state_since = now
                    speak_guidance(f"{target_eye} eye ready. Hold steady.")
            else:
                if global_state not in (STATE_DISPENSING, STATE_VERIFYING, STATE_COOLDOWN):
                    global_state = STATE_ALIGNING

            # If READY for enough time, "dispense"
            if global_state == STATE_READY and (time.time() - state_since) >= STABLE_HOLD_SECONDS:
                global_state = STATE_DISPENSING
                state_since = time.time()
                speak_guidance("Perfect position. Dispensing now.")

                # Build eye ROI for verification
                if eye_pts_hull and len(eye_pts_hull) >= 3:
                    xs = [p[0] for p in eye_pts_hull]; ys = [p[1] for p in eye_pts_hull]
                    pad = 8
                    x1, x2 = max(min(xs)-pad, 0), min(max(xs)+pad, W)
                    y1, y2 = max(min(ys)-pad, 0), min(max(ys)+pad, H)
                else:
                    x1,y1,x2,y2 = max(cx-60,0), max(cy-40,0), min(cx+60,W), min(cy+40,H)

                roi_before = frame[y1:y2, x1:x2].copy()
                roi_before_gray = cv2.cvtColor(roi_before, cv2.COLOR_BGR2GRAY) if roi_before.size else None

                # small travel time
                time.sleep(0.45)
                ok2, frame2 = cap.read()
                if not ok2:
                    frame2 = frame.copy()
                roi_after = frame2[y1:y2, x1:x2].copy()
                roi_after_gray = cv2.cvtColor(roi_after, cv2.COLOR_BGR2GRAY) if roi_after.size else None

                # Motion score
                diff_score = mean_abs_diff(roi_before_gray, roi_after_gray)

                # Extra: detect motion entering from top region of ROI
                motion_ok = False
                if roi_before_gray is not None and roi_after_gray is not None:
                    diff = cv2.absdiff(roi_before_gray, roi_after_gray)
                    _, th = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    th = cv2.medianBlur(th, 3)
                    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    h_roi = roi_after_gray.shape[0]
                    top_band = int(h_roi * ENTRY_FROM_ABOVE_BIAS)
                    for c in contours:
                        area = cv2.contourArea(c)
                        if area < DROP_MIN_AREA:
                            continue
                        x,y,wc,hc = cv2.boundingRect(c)
                        # consider "from above" if top of bbox is in top band
                        if y <= top_band:
                            motion_ok = True
                            break

                drop_detected = (diff_score > DROP_DIFF_THRESH) and motion_ok

                if drop_detected:
                    speak_guidance("Drop detected successfully.")
                    log_event(target_eye, True)
                else:
                    speak_guidance("No drop detected. Please try again.")
                    log_event(target_eye, False)

                target_state.last_dispense = time.time()
                global_state = STATE_COOLDOWN
                state_since = time.time()

            # Draw HUD
            cv2.putText(frame, f"Target: {target_eye}", (10,22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            cv2.putText(frame, f"EAR: {ear:.3f}  Open:{int(eye_open)}", (10,46), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
            cv2.putText(frame, f"State: {global_state}", (10,70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
            cv2.putText(frame, f"Guide: {guidance}", (10,H-14), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,200,0), 2)

        else:
            guidance = "No face detected"
            global_state = STATE_ALIGNING
            cv2.putText(frame, guidance, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)

        # Speak guidance updates
        speak_guidance(guidance)

        # Show video
        cv2.imshow("Smart Eye Drop Dispenser (Final)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    tts.stop()
    cap.release()
    cv2.destroyAllWindows()
 # type: ignore