import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import pygame
import time
import os

# ---------------- PATH SETUP ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALERT_PATH = os.path.join(BASE_DIR, "assets", "alert.wav")
MODEL_PATH = os.path.join(
    BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat"
)

# ---------------- AUDIO INIT ----------------
pygame.mixer.init()
alert_sound = pygame.mixer.Sound(ALERT_PATH)

def play_alert():
    alert_sound.play()

last_alert_time = 0

# ---------------- LOAD MODELS ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# ---------------- EAR FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------- BLINK & PERCLOS SETUP ----------------
EYE_AR_THRESH = 0.25

blink_count = 0
eye_closed = False
closed_eye_frames = 0
total_frames = 0
start_time = time.time()

# ---------------- ALERT ESCALATION SETUP (NEW) ----------------
drowsy_start_time = None

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(0)
fatigue_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    total_frames += 1

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        # -------- Blink & PERCLOS Logic --------
        if ear < EYE_AR_THRESH:
            closed_eye_frames += 1
            if not eye_closed:
                eye_closed = True
        else:
            if eye_closed:
                blink_count += 1
                eye_closed = False

        elapsed_time = time.time() - start_time
        blink_rate = (blink_count / elapsed_time) * 60 if elapsed_time > 0 else 0
        perclos = (closed_eye_frames / total_frames) * 100

        # -------- Existing Fatigue Logic (UNCHANGED) --------
        if ear < 0.25:
            fatigue_score += 1
        else:
            fatigue_score = max(0, fatigue_score - 1)

        fatigue_confidence = min(100, int((fatigue_score / 25) * 100))

        # ---------------- DISPLAY (UNCHANGED) ----------------
        cv2.putText(frame, f"Fatigue: {fatigue_score}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)

        cv2.putText(frame, f"Fatigue Risk: {fatigue_confidence}%",
                    (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.putText(frame, f"Blinks/min: {int(blink_rate)}",
                    (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        cv2.putText(frame, f"PERCLOS: {int(perclos)}%",
                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

        # ---------------- ALERT ESCALATION LOGIC (ONLY NEW PART) ----------------
        if fatigue_score > 20:
            if drowsy_start_time is None:
                drowsy_start_time = time.time()

            drowsy_duration = time.time() - drowsy_start_time

            if drowsy_duration < 10:
                cooldown = 4
                alert_text = "DROWSY!"
            elif drowsy_duration < 20:
                cooldown = 2
                alert_text = "TAKE A BREAK"
            else:
                cooldown = 1
                alert_text = "DANGER!"

            cv2.putText(frame, alert_text,
                        (50, 160), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)

            if time.time() - last_alert_time > cooldown:
                play_alert()
                last_alert_time = time.time()
        else:
            drowsy_start_time = None

    cv2.imshow("SmartVision", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()