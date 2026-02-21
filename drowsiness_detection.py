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
ALERT_COOLDOWN = 3  # seconds

# ---------------- LOAD MODELS ----------------
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(MODEL_PATH)

# ---------------- EAR FUNCTION ----------------
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# ---------------- VIDEO ----------------
cap = cv2.VideoCapture(0)
fatigue_score = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)]
        right_eye = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)]

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < 0.25:
            fatigue_score += 1
        else:
            fatigue_score = max(0, fatigue_score - 1)

        cv2.putText(
            frame, f"Fatigue: {fatigue_score}",
            (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
        )

        if fatigue_score > 20:
            cv2.putText(
                frame, "DROWSY!",
                (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3
            )

            if time.time() - last_alert_time > ALERT_COOLDOWN:
                play_alert()
                last_alert_time = time.time()

    cv2.imshow("SmartVision", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()