# code_final.py
# Smart Eye Drop Dispenser (final corrected)
# Put best.pt in same folder if you want YOLO-based dropper detection.
# Press 'q' to quit.

import os, sys, time, csv, threading, tempfile, subprocess, json
from queue import Queue
from datetime import datetime, date

try:
    import cv2, numpy as np, mediapipe as mp
except Exception as e:
    print("Missing dependency:", e)
    raise

# Optional YOLO
YOLO_AVAILABLE = False
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False

# -------------------- SETTINGS --------------------
MODEL_PATH = "best.pt"           # local YOLO model (optional)
CAPTURE_INDEX = 0
IMG_SIZE = 640
CONF_THRESH = 0.25

EAR_OPEN_THRESH = 0.22
EMA_ALPHA = 0.35

ALIGN_PIX_THRESHOLD = 36
DEBOUNCE_FRAMES = 6
STABLE_HOLD_SECONDS = 0.7

DROP_DIFF_THRESH = 12.0
DROP_MIN_AREA = 10
ENTRY_FROM_ABOVE_BIAS = 0.14

PER_EYE_COOLDOWN = 4.0

WINDOW_NAME = "Smart Eye Drop Dispenser"
LOGFILE = "drop_log.csv"

GUIDANCE_REPEAT_SEC = 1.6
TTS_COOLDOWN_GENERAL = 0.8
# ---------------------------------------------------

LOCKFILE = os.path.join(tempfile.gettempdir(), "smart_drop_lock.txt")

def create_lock_or_exit():
    mypid = os.getpid()
    if os.path.exists(LOCKFILE):
        try:
            with open(LOCKFILE,"r") as f:
                data = f.read().strip()
            other = int(data.split("|")[0])
            if other != mypid:
                try:
                    if os.name == "nt":
                        import ctypes
                        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
                        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, other)
                        if handle:
                            ctypes.windll.kernel32.CloseHandle(handle)
                            print(f"Another instance detected (PID {other}). Exiting.")
                            sys.exit(0)
                    else:
                        os.kill(other, 0)
                        print(f"Another instance detected (PID {other}). Exiting.")
                        sys.exit(0)
                except Exception:
                    os.remove(LOCKFILE)
        except Exception:
            try: os.remove(LOCKFILE)
            except: pass
    with open(LOCKFILE,"w") as f:
        f.write(f"{mypid}|{datetime.now().isoformat()}|{os.path.abspath(__file__)}")

def remove_lock():
    try:
        if os.path.exists(LOCKFILE):
            os.remove(LOCKFILE)
    except: pass

def ensure_log():
    if not os.path.exists(LOGFILE):
        with open(LOGFILE,"w",newline="") as f:
            csv.writer(f).writerow(["timestamp","date","event","eye","success"])

def log_event(event, eye=None, success=None):
    with open(LOGFILE,"a",newline="") as f:
        csv.writer(f).writerow([datetime.now().isoformat(), date.today().isoformat(), event, "" if eye is None else eye, "" if success is None else int(bool(success))])

class TTSWorker(threading.Thread):
    def __init__(self, cooldown=TTS_COOLDOWN_GENERAL):
        super().__init__(daemon=True)
        self.q = Queue()
        self.cooldown = cooldown
        self._last_event = 0.0
        self.backend = None
        self.engine = None
        self._running = True
        self._init_try = False
        self.start()

    def _init_backend(self):
        if self._init_try:
            return
        self._init_try = True
        if os.name == "nt":
            try:
                import win32com.client
                self.backend = "sapi"
                self.engine = win32com.client.Dispatch("SAPI.SpVoice")
                print("[TTS] Using SAPI backend")
                return
            except Exception:
                self.engine = None
        try:
            import pyttsx3
            self.backend = "pyttsx3"
            self.engine = pyttsx3.init()
            try:
                rate = self.engine.getProperty("rate")
                self.engine.setProperty("rate", max(120, rate - 20))
            except Exception:
                pass
            print("[TTS] Using pyttsx3 backend")
            return
        except Exception:
            self.engine = None
        if os.name == "nt":
            self.backend = "powershell"
            print("[TTS] Using PowerShell fallback")
        else:
            self.backend = "print"
            print("[TTS] No TTS backend; falling back to printing")

    def _sapi_speak(self, text):
        try:
            self.engine.Speak(text)
        except Exception as e:
            print("[TTS] SAPI speak failed:", e)

    def _pyttsx3_speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print("[TTS] pyttsx3 speak failed:", e)

    def _powershell_speak(self, text):
        # Use json.dumps to safely quote/escape the text for PowerShell
        safe_text = json.dumps(text)  # produces a double-quoted JSON string with proper escaping
        cmd = f'Add-Type -AssemblyName System.Speech; (New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak({safe_text});'
        try:
            subprocess.run(["powershell", "-Command", cmd], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        except Exception as e:
            print("[TTS] PowerShell speak failed:", e)

    def run(self):
        self._init_backend()
        while self._running:
            item = self.q.get()
            if item is None:
                break
            text, bypass = item
            try:
                if self.backend == "sapi":
                    self._sapi_speak(text)
                elif self.backend == "pyttsx3":
                    self._pyttsx3_speak(text)
                elif self.backend == "powershell":
                    self._powershell_speak(text)
                else:
                    print("[TTS]", text)
            except Exception as e:
                print("[TTS] Exception:", e)

    def say_immediate(self, text):
        self.q.put((text, True))
        self._last_event = time.time()

    def say_event(self, text):
        now = time.time()
        if now - self._last_event >= self.cooldown:
            self.q.put((text, False))
            self._last_event = now

    def stop(self):
        self._running = False
        try:
            self.q.put(None)
        except: pass

class EMA:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = float(alpha)
        self.v = None
    def update(self, x):
        if x is None:
            return None
        if isinstance(x, (list, tuple, np.ndarray)):
            arr = np.array(x, dtype=float)
            if arr.ndim == 1 and arr.size == 2:
                if self.v is None:
                    self.v = arr
                else:
                    self.v = self.alpha*arr + (1-self.alpha)*self.v
                return (float(self.v[0]), float(self.v[1]))
            arr2 = arr.reshape(-1,2).mean(axis=0)
            if self.v is None:
                self.v = arr2
            else:
                self.v = self.alpha*arr2 + (1-self.alpha)*self.v
            return (float(self.v[0]), float(self.v[1]))
        else:
            val = float(x)
            if self.v is None:
                self.v = val
            else:
                self.v = self.alpha*val + (1-self.alpha)*self.v
            return float(self.v)
    def value(self):
        if self.v is None:
            return None
        if isinstance(self.v, np.ndarray) and self.v.size==2:
            return (float(self.v[0]), float(self.v[1]))
        return float(self.v)

mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                             min_detection_confidence=0.45, min_tracking_confidence=0.45)
LEFT_EAR = [33,160,158,133,153,144]
RIGHT_EAR= [362,385,387,263,373,380]

def get_eye_points(lms, idxs, W, H):
    pts = []
    for i in idxs:
        if i < len(lms):
            p = lms[i]
            pts.append((int(p.x*W), int(p.y*H)))
    return pts if len(pts)==len(idxs) else None

def euclid(a,b):
    return float(np.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1])))

def compute_ear(pts):
    p1,p2,p3,p4,p5,p6 = pts
    num = euclid(p2,p6) + euclid(p3,p5)
    den = 2.0 * euclid(p1,p4) + 1e-9
    return num/den

def draw_text_box(img, text, org, scale=0.7, padding=8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (w,h), baseline = cv2.getTextSize(text, font, scale, 2)
    x,y = int(org[0]), int(org[1])
    x1 = max(0, x - padding)
    y1 = max(0, y - h - padding)
    x2 = min(img.shape[1], x + w + padding)
    y2 = min(img.shape[0], y + padding + 2)
    cv2.rectangle(img, (x1,y1), (x2,y2), (255,255,255), -1, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), font, scale, (0,0,0), 2, cv2.LINE_AA)

yolo_model = None
if YOLO_AVAILABLE and os.path.exists(MODEL_PATH):
    try:
        yolo_model = YOLO(MODEL_PATH)
        print("[YOLO] Loaded model:", MODEL_PATH)
    except Exception as e:
        print("[YOLO] Load failed:", e)
else:
    if YOLO_AVAILABLE:
        print("[YOLO] Model path not found:", MODEL_PATH)
    else:
        print("[YOLO] ultralytics not installed; continuing without YOLO")

def main():
    create_lock_or_exit()
    ensure_log()
    log_event("session_start")
    tts = TTSWorker()
    tts.say_event("Starting eye drop dispenser. Please face the camera.")

    cap = cv2.VideoCapture(CAPTURE_INDEX, cv2.CAP_DSHOW if hasattr(cv2,"CAP_DSHOW") else 0)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAPTURE_INDEX)
    time.sleep(0.1)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    class EyeState:
        def __init__(self):
            self.center_ema = EMA()
            self.ear_ema = EMA()
            self.debounce = 0
            self.last_dispense = 0.0

    left = EyeState(); right = EyeState()
    target_eye = "left"
    global_state = "ALIGN"
    state_since = time.time()

    last_guidance_time = 0.0
    last_guidance_text = ""

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            H, W = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            lms = res.multi_face_landmarks[0].landmark if res and res.multi_face_landmarks else None

            left6 = right6 = None
            if lms:
                left6 = get_eye_points(lms, LEFT_EAR, W, H)
                right6 = get_eye_points(lms, RIGHT_EAR, W, H)

            if left6:
                left.center_ema.update(((left6[0][0]+left6[3][0])/2.0, (left6[0][1]+left6[3][1])/2.0))
                try:
                    left_ear_raw = compute_ear(left6)
                except:
                    left_ear_raw = 0.0
                left.ear_ema.update(left_ear_raw)
                cv2.polylines(frame, [np.array(left6, np.int32)], True, (0,200,0), 1)

            if right6:
                right.center_ema.update(((right6[0][0]+right6[3][0])/2.0, (right6[0][1]+right6[3][1])/2.0))
                try:
                    right_ear_raw = compute_ear(right6)
                except:
                    right_ear_raw = 0.0
                right.ear_ema.update(right_ear_raw)
                cv2.polylines(frame, [np.array(right6, np.int32)], True, (0,200,0), 1)

            detections = []
            if yolo_model is not None:
                try:
                    preds = yolo_model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)
                    for r in preds:
                        names = r.names if hasattr(r,"names") else {}
                        boxes = getattr(r, "boxes", None)
                        if boxes is None: continue
                        for b in boxes:
                            cls = int(b.cls[0]) if hasattr(b.cls, "__len__") else int(b.cls)
                            label = names.get(cls, str(cls)).lower()
                            if label in ("eyedrop", "dropper", "mouth"):
                                xyxy = b.xyxy[0]
                                xy = xyxy.cpu().numpy() if hasattr(xyxy, "cpu") else np.array(xyxy)
                                x1,y1,x2,y2 = [int(v) for v in xy]
                                cx,cy = (x1+x2)//2, (y1+y2)//2
                                detections.append({"label":"dropper","box":(x1,y1,x2,y2),"center":(float(cx), float(cy))})
                                cv2.rectangle(frame, (x1,y1),(x2,y2), (180,80,200), 2)
                                draw_text_box(frame, "Dropper", (x1, max(18, y1-6)), scale=0.55)
                except Exception as e:
                    print("[YOLO] predict error:", e)
                    pass

            lcen = left.center_ema.value()
            rcen = right.center_ema.value()
            def near_center(pt):
                return euclid(pt, (W/2.0, H/2.0))
            if lcen and rcen:
                target_eye = "left" if near_center(lcen) <= near_center(rcen) else "right"
            elif lcen:
                target_eye = "left"
            elif rcen:
                target_eye = "right"
            else:
                target_eye = None

            guidance = ""
            now = time.time()

            if target_eye is None:
                guidance = "Place your face in view"
                global_state = "ALIGN"
                if now - last_guidance_time > 1.2 or last_guidance_text != guidance:
                    tts.say_immediate(guidance)
                    last_guidance_time = now; last_guidance_text = guidance
            else:
                t = left if target_eye=="left" else right
                center = t.center_ema.value()
                ear = t.ear_ema.value() or 0.0

                chosen_drop = None
                if detections and center:
                    best = None
                    for d in detections:
                        dist = euclid(d["center"], center)
                        if best is None or dist < best[0]:
                            best = (dist, d)
                    if best:
                        chosen_drop = best[1]

                if chosen_drop:
                    dx = chosen_drop["center"][0] - center[0]
                    dy = chosen_drop["center"][1] - center[1]
                    if abs(dx) > ALIGN_PIX_THRESHOLD and abs(dx) >= abs(dy):
                        guidance = "Move dropper " + ("left" if dx < 0 else "right")
                    elif abs(dy) > ALIGN_PIX_THRESHOLD and abs(dy) > abs(dx):
                        guidance = "Move dropper " + ("up" if dy < 0 else "down")
                    else:
                        if ear < EAR_OPEN_THRESH:
                            guidance = "Please open your eye"
                        else:
                            guidance = "Good alignment. Hold steady."
                else:
                    if center:
                        fx, fy = center
                        dx = fx - W/2.0
                        dy = fy - H/2.0
                        if abs(dx) > ALIGN_PIX_THRESHOLD and abs(dx) >= abs(dy):
                            guidance = "Move your face " + ("left" if dx < 0 else "right")
                        elif abs(dy) > ALIGN_PIX_THRESHOLD:
                            guidance = "Move your face " + ("up" if dy < 0 else "down")
                        else:
                            if ear < EAR_OPEN_THRESH:
                                guidance = "Please open your eye"
                            else:
                                guidance = "Good alignment. Hold steady."
                    else:
                        guidance = "Bring face to camera"

                if guidance.startswith("Good alignment") and ear >= EAR_OPEN_THRESH:
                    t.debounce += 1
                else:
                    t.debounce = 0

                if t.debounce >= DEBOUNCE_FRAMES and (now - t.last_dispense) > PER_EYE_COOLDOWN:
                    if global_state != "READY":
                        global_state = "READY"; state_since = now
                        tts.say_event(f"{target_eye} eye ready. Hold steady.")
                        log_event("eye_ready", eye=target_eye)
                    else:
                        if (now - state_since) >= STABLE_HOLD_SECONDS:
                            global_state = "DISPENSING"; state_since = now
                            tts.say_event("Dispensing now.")
                            log_event("dispense_start", eye=target_eye)
                if global_state == "DISPENSING":
                    if center is None:
                        # fallback small ROI center of frame
                        ex,ey = int(W/2), int(H/2)
                    else:
                        ex,ey = int(center[0]), int(center[1])
                    R = 48
                    x1 = max(0, ex - R); y1 = max(0, ey - R)
                    x2 = min(W - 1, ex + R); y2 = min(H - 1, ey + R)
                    frames_gray = []
                    for _ in range(3):
                        t_ok, f2 = cap.read()
                        if not t_ok: break
                        f2 = cv2.flip(f2, 1)
                        roi = f2[y1:y2, x1:x2].copy()
                        if roi.size == 0:
                            frames_gray.append(None)
                        else:
                            frames_gray.append(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))
                        time.sleep(0.12)
                    diff_score = 0.0
                    motion_ok = False
                    comb = None
                    for i in range(len(frames_gray)-1):
                        a = frames_gray[i]; b = frames_gray[i+1]
                        if a is None or b is None: continue
                        d = cv2.absdiff(a, b)
                        comb = d if comb is None else cv2.max(comb, d)
                    if comb is not None:
                        diff_score = float(np.mean(comb))
                        _, th = cv2.threshold(comb, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
                        th = cv2.medianBlur(th, 3)
                        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        top_band = int(comb.shape[0] * ENTRY_FROM_ABOVE_BIAS)
                        for c in contours:
                            if cv2.contourArea(c) < DROP_MIN_AREA: continue
                            x,y,wc,hc = cv2.boundingRect(c)
                            if y <= top_band + 6:
                                motion_ok = True
                                break
                    drop_detected = (diff_score > DROP_DIFF_THRESH) and motion_ok
                    if drop_detected:
                        tts.say_immediate("Drop detected successfully.")
                        log_event("drop_detected", eye=target_eye, success=True)
                    else:
                        tts.say_immediate("No drop detected. Please try again.")
                        log_event("drop_detected", eye=target_eye, success=False)
                    t.last_dispense = time.time()
                    global_state = "VERIFY"
                    state_since = time.time()

                elif global_state == "VERIFY":
                    if (time.time() - state_since) > 0.7:
                        global_state = "COOLDOWN"
                        state_since = time.time()
                        target_eye = "right" if target_eye == "left" else "left"
                        tts.say_immediate(f"Now move to the {target_eye} eye.")
                        log_event("switch_eye", eye=target_eye)

                elif global_state == "COOLDOWN":
                    guidance = "Cooldown..."
                    if (time.time() - state_since) > PER_EYE_COOLDOWN:
                        global_state = "ALIGN"
                        state_since = time.time()

            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (W, 110), (255,255,255), -1)
            alpha = 0.6
            frame = cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0)

            draw_text_box(frame, f"Target: {target_eye if target_eye else 'None'}", (12, 30), scale=0.86)
            draw_text_box(frame, f"State: {global_state}", (12, 62), scale=0.58)
            draw_text_box(frame, f"Guidance: {guidance}", (12, 94), scale=0.52)

            if lcen:
                cv2.circle(frame, (int(lcen[0]), int(lcen[1])), 3, (0,150,255), -1)
                cv2.putText(frame, "L", (int(lcen[0])-10, int(lcen[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)
            if rcen:
                cv2.circle(frame, (int(rcen[0]), int(rcen[1])), 3, (0,150,255), -1)
                cv2.putText(frame, "R", (int(rcen[0])-10, int(rcen[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,150,255), 2)

            now = time.time()
            if guidance:
                speak_it = False
                if guidance != last_guidance_text:
                    speak_it = True
                elif (now - last_guidance_time) >= GUIDANCE_REPEAT_SEC:
                    speak_it = True
                if speak_it:
                    tts.say_immediate(guidance)
                    last_guidance_text = guidance
                    last_guidance_time = now

            cv2.imshow(WINDOW_NAME, frame)
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                log_event("user_quit")
                break

    finally:
        try:
            tts.stop()
        except: pass
        try:
            cap.release()
            cv2.destroyAllWindows()
        except: pass
        remove_lock()
        log_event("session_end")
        print("Exited cleanly. Log saved to", LOGFILE)

if __name__ == "__main__":
    main()
