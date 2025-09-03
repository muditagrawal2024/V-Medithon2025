import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import time
import csv
import os
from datetime import datetime

# ===============================
# TTS SETUP
# ===============================
ttl = pyttsx3.init()
ttl.setProperty("rate", 150)   # slower speech
ttl.setProperty("volume", 1.0)

last_spoken = None
def speak(text):
    """Speak only if new instruction"""
    global last_spoken
    if text != last_spoken:
        print("VOICE:", text)
        ttl.say(text)
        ttl.runAndWait()
        last_spoken = text

# ===============================
# LOGGING
# ===============================
LOG_FILE = "drop_log.csv"
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "event"])

def log_event(event):
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), event])
    print("LOG:", event)

# ===============================
# FACE + EYE DETECTION
# ===============================
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(refine_landmarks=True)

def get_eye_centers(landmarks, w, h):
    left_ids = [33, 133]   # left eye corners
    right_ids = [362, 263] # right eye corners
    lx = int(np.mean([landmarks[i].x for i in left_ids]) * w)
    ly = int(np.mean([landmarks[i].y for i in left_ids]) * h)
    rx = int(np.mean([landmarks[i].x for i in right_ids]) * w)
    ry = int(np.mean([landmarks[i].y for i in right_ids]) * h)
    return (lx, ly), (rx, ry)

# ===============================
# DRAWING UTILITIES
# ===============================
def draw_text(frame, text, pos):
    cv2.putText(frame, text, pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,0), 3, cv2.LINE_AA)

def draw_dropper(frame, dropper_pos, dispensing=False, droplet_y=None):
    x, y = dropper_pos
    cv2.line(frame, (x, y-40), (x, y+20), (255,0,0), 6)  # stem
    cv2.circle(frame, (x, y-50), 15, (200,200,255), -1)  # bulb
    if dispensing and droplet_y is not None:  # droplet
        cv2.circle(frame, (x, droplet_y), 10, (255,0,255), -1)

# ===============================
# MAIN LOOP
# ===============================
cap = cv2.VideoCapture(0)
W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
center = (W//2, H//2)

state = "ALIGN_LEFT"
dropper_pos = [center[0], H+50]
dispense_timer = 0
droplet_y = None
current_eye = "left"

speak("Starting smart eye drop dispenser. Please align your face.")
log_event("system_start")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark
        left_eye, right_eye = get_eye_centers(landmarks, W, H)

        cv2.circle(frame, left_eye, 5, (0,255,0), -1)
        cv2.circle(frame, right_eye, 5, (0,255,0), -1)

        target = left_eye if current_eye == "left" else right_eye
        dx, dy = target[0]-center[0], target[1]-center[1]

        # ================= STATE MACHINE =================
        if state.startswith("ALIGN"):
            if abs(dx) > 40:
                msg = "Please move your face left" if dx > 0 else "Please move your face right"
                draw_text(frame, msg, (50,50))
                speak(msg)
                log_event(msg)
            elif abs(dy) > 40:
                msg = "Please move your face up" if dy > 0 else "Please move your face down"
                draw_text(frame, msg, (50,50))
                speak(msg)
                log_event(msg)
            else:
                msg = "Aligned. Bringing dropper."
                draw_text(frame, msg, (50,50))
                speak(msg)
                log_event("aligned_"+current_eye)
                state = "MOVE_DROPPER"

        elif state == "MOVE_DROPPER":
            if dropper_pos[1] > target[1]-80:
                dropper_pos[1] -= 3  # slower
                speak("Moving dropper into position")
            else:
                state = "DISPENSE"
                dispense_timer = time.time()
                droplet_y = dropper_pos[1]
                speak("Dispensing drop now")
                log_event("dispense_"+current_eye)

        elif state == "DISPENSE":
            if droplet_y < target[1]+20:
                droplet_y += 3
            draw_dropper(frame, dropper_pos, dispensing=True, droplet_y=droplet_y)

            if time.time() - dispense_timer > 3:
                if current_eye == "left":
                    current_eye = "right"
                    state = "MOVE_SIDE"
                    speak("Now moving dropper towards your right eye")
                    log_event("move_side")
                else:
                    state = "DONE"
                    speak("Process complete. Please close your eyes.")
                    log_event("done")

        elif state == "MOVE_SIDE":
            if abs(dropper_pos[0] - right_eye[0]) > 5:
                step = 4 if dropper_pos[0] < right_eye[0] else -4
                dropper_pos[0] += step
                speak("Moving dropper sideways to right eye")
            else:
                dropper_pos[0] = right_eye[0]
                dropper_pos[1] = H+50
                state = "ALIGN_RIGHT"
                speak("Now align your right eye with the camera")
                log_event("next_eye_right")

        elif state == "DONE":
            draw_text(frame, "Done. Thank you!", (50,50))

        if state in ["MOVE_DROPPER","DISPENSE","MOVE_SIDE"]:
            draw_dropper(frame, dropper_pos, dispensing=(state=="DISPENSE"), droplet_y=droplet_y)

    else:
        draw_text(frame, "Place your face in view", (50,50))
        speak("Place your face in view")
        log_event("no_face")

    cv2.imshow("Smart Eye Drop", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
speak("Exited cleanly")
log_event("session_end")
