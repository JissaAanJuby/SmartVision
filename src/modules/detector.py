import time
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from .metrics import calculate_ear, calculate_mar
from .pose import estimate_head_pose

ALERT_AUDIO_SRC = "/assets/alert.wav"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [13, 14, 61, 291]


class FatigueDetector:
    """
    State machine notes:

    - Thresholds are wall-clock durations (seconds), not frame counts.
      Frame counts only mean something at a fixed FPS; this pipeline's
      effective rate varies with hardware load (single-worker thread
      pool, model inference time), so duration-based checks are the
      only way to get consistent DROWSY/DANGER timing across machines.
    - EAR/MAR/pose values are exponentially smoothed before being
      compared against thresholds, so a single noisy landmark frame
      (camera jitter, partial occlusion) can't flip the state on its
      own — it has to persist.
    - Eye closure, yawning, AND head pose each have their own duration
      tracker and can independently escalate to DANGER. Previously,
      extreme head tilt could only ever nudge the state to DROWSY.
    - running_mode=VIDEO (not the default IMAGE) tells MediaPipe these
      frames are a continuous sequence, so it tracks the face between
      frames instead of re-running full detection on every single one.
      This is the single biggest speed win available here — IMAGE mode
      pays full detection cost every frame regardless of resolution.
    - Inference runs on a downscaled copy of the frame (landmark
      coordinates from MediaPipe are normalized 0-1, so they still map
      correctly onto the full-resolution frame for drawing/streaming).
    - A dummy frame is pushed through the graph once at startup so the
      one-time TFLite/graph initialization cost is paid here, not on
      the first real camera frame the user sees.
    """

    INFER_WIDTH = 320  # frame is downscaled to this width before detection

    def __init__(self, model_path: str, logger=None):
        base_options = python.BaseOptions(model_asset_path=str(model_path))
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)
        self.logger = logger  # FatigueLogger instance or None

        self._start_time = time.monotonic()
        self._last_timestamp_ms = -1

        # Warm-up: forces lazy model/graph initialization to happen now,
        # not on the first real frame from the camera.
        dummy = np.zeros((self.INFER_WIDTH, self.INFER_WIDTH, 3), dtype=np.uint8)
        dummy_mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=dummy)
        self.landmarker.detect_for_video(dummy_mp_image, self._next_timestamp_ms())

        # --- Value thresholds ---
        self.EYE_AR_THRESH = 0.21
        self.MAR_YAWN_THRESH = 0.6
        self.PITCH_DROWSY_DEG = 15.0
        self.YAW_DROWSY_DEG = 22.0
        self.PITCH_DANGER_DEG = 25.0
        self.YAW_DANGER_DEG = 35.0

        # --- Duration thresholds (seconds) ---
        self.BLINK_MIN_SEC = 0.08      # below this = landmark/camera noise, not a real blink
        self.DROWSY_EYE_SEC = 1.0      # sustained closure -> DROWSY
        self.DANGER_EYE_SEC = 2.5      # microsleep -> DANGER
        self.DROWSY_YAWN_SEC = 1.2
        self.DANGER_YAWN_SEC = 3.0
        self.DROWSY_POSE_SEC = 1.2
        self.DANGER_POSE_SEC = 2.5

        # --- EMA smoothing factors (higher = more responsive, less smooth) ---
        self.EAR_ALPHA = 0.4
        self.MAR_ALPHA = 0.4
        self.POSE_ALPHA = 0.3

        # --- Smoothed running values ---
        self.ema_ear = None
        self.ema_mar = None
        self.ema_pitch = 0.0
        self.ema_yaw = 0.0
        self._last_valid_pitch = 0.0
        self._last_valid_yaw = 0.0

        # --- Duration trackers (monotonic start-timestamps, None = not active) ---
        self._eye_closed_since = None
        self._mouth_open_since = None
        self._pose_bad_since = None

        # --- Counters / state ---
        self.eye_closed = False
        self.blink_count = 0
        self.is_yawning = False
        self.yawn_count = 0
        self.fatigue_score = 0.0
        self.state = "NORMAL"
        self._last_logged_state = "NORMAL"

        # --- FPS tracking (for on-screen/telemetry visibility only) ---
        self._last_frame_end = None
        self.processing_fps = 0.0

    # ---- internal helpers --------------------------------------------
    def _next_timestamp_ms(self) -> int:
        """detect_for_video requires strictly increasing millisecond timestamps."""
        ts = int((time.monotonic() - self._start_time) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    @staticmethod
    def _ema(prev, new, alpha):
        return new if prev is None else (alpha * new + (1 - alpha) * prev)

    def _track_duration(self, condition: bool, since_attr: str) -> float:
        """How long `condition` has been continuously true, in seconds."""
        now = time.monotonic()
        since = getattr(self, since_attr)
        if condition:
            if since is None:
                setattr(self, since_attr, now)
                return 0.0
            return now - since
        setattr(self, since_attr, None)
        return 0.0

    def _evaluate_state(self, eye_closed_dur, yawn_dur, pose_bad_dur) -> str:
        if (eye_closed_dur >= self.DANGER_EYE_SEC
                or yawn_dur >= self.DANGER_YAWN_SEC
                or pose_bad_dur >= self.DANGER_POSE_SEC):
            return "DANGER"
        if (eye_closed_dur >= self.DROWSY_EYE_SEC
                or yawn_dur >= self.DROWSY_YAWN_SEC
                or pose_bad_dur >= self.DROWSY_POSE_SEC):
            return "DROWSY"
        return "NORMAL"

    # ---- main entry point ----------------------------------------------
    def process_frame(self, frame: np.ndarray):
        frame_start = time.monotonic()
        h, w, _ = frame.shape

        # Detection runs on a downscaled copy — landmark coordinates come back
        # normalized (0-1), so they still map correctly onto the full-res
        # frame below. This is the main lever on face-detection cost; the
        # landmark stage itself already works on a fixed-size crop.
        if w > self.INFER_WIDTH:
            scale = self.INFER_WIDTH / w
            infer_frame = cv2.resize(frame, (self.INFER_WIDTH, max(1, int(h * scale))))
        else:
            infer_frame = frame

        rgb_frame = cv2.cvtColor(infer_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect_for_video(mp_image, self._next_timestamp_ms())

        if self._last_frame_end is not None:
            dt = frame_start - self._last_frame_end
            if dt > 0:
                inst_fps = 1.0 / dt
                self.processing_fps = self._ema(self.processing_fps or None, inst_fps, 0.2)
        self._last_frame_end = frame_start

        metrics = {
            "ear": 0.0, "mar": 0.0, "pitch": 0, "yaw": 0,
            "blink_count": self.blink_count, "yawn_count": self.yawn_count,
            "is_yawning": self.is_yawning,
            "fatigue_score": round(self.fatigue_score, 1),
            "fatigue_confidence": int(min(100, (self.fatigue_score / 30.0) * 100)),
            "alert": self.state, "state": self.state,
            "audio_alert": False, "audio_src": ALERT_AUDIO_SRC,
            "state_changed": False,
            "fps": round(self.processing_fps, 1),
        }

        if not detection_result.face_landmarks:
            return frame, metrics

        landmarks = detection_result.face_landmarks[0]
        left_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in LEFT_EYE]
        right_eye = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in RIGHT_EYE]
        mouth = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in MOUTH]

        raw_ear = (calculate_ear(left_eye) + calculate_ear(right_eye)) / 2.0
        raw_mar = calculate_mar(mouth)
        pitch, yaw, pose_ok = estimate_head_pose(landmarks, w, h)
        if pose_ok:
            self._last_valid_pitch, self._last_valid_yaw = pitch, yaw
        else:
            pitch, yaw = self._last_valid_pitch, self._last_valid_yaw

        # Smoothing absorbs single-frame jitter before it ever reaches a threshold check.
        self.ema_ear = self._ema(self.ema_ear, raw_ear, self.EAR_ALPHA)
        self.ema_mar = self._ema(self.ema_mar, raw_mar, self.MAR_ALPHA)
        self.ema_pitch = self._ema(self.ema_pitch, pitch, self.POSE_ALPHA)
        self.ema_yaw = self._ema(self.ema_yaw, yaw, self.POSE_ALPHA)
        ear, mar = self.ema_ear, self.ema_mar

        # --- Blink counting (independent of drowsiness thresholds) ---
        eyes_closed_now = ear < self.EYE_AR_THRESH
        closed_dur = self._track_duration(eyes_closed_now, "_eye_closed_since")
        if eyes_closed_now and closed_dur >= self.BLINK_MIN_SEC and not self.eye_closed:
            self.eye_closed = True
        elif not eyes_closed_now and self.eye_closed:
            self.blink_count += 1
            self.eye_closed = False

        # --- Yawn counting ---
        mouth_open_now = mar > self.MAR_YAWN_THRESH
        yawn_dur = self._track_duration(mouth_open_now, "_mouth_open_since")
        if mouth_open_now and not self.is_yawning:
            self.is_yawning = True
            self.yawn_count += 1
            self.fatigue_score = min(100.0, self.fatigue_score + 4.0)
        elif not mouth_open_now:
            self.is_yawning = False

        # --- Head pose persistence (can independently escalate to DANGER) ---
        pose_bad_now = abs(self.ema_pitch) > self.PITCH_DROWSY_DEG or abs(self.ema_yaw) > self.YAW_DROWSY_DEG
        pose_bad_dur = self._track_duration(pose_bad_now, "_pose_bad_since")

        state = self._evaluate_state(closed_dur, yawn_dur, pose_bad_dur)
        self.state = state

        if state == "DANGER":
            self.fatigue_score = min(100.0, self.fatigue_score + 2.0)
        elif state == "DROWSY":
            self.fatigue_score = min(100.0, self.fatigue_score + 1.0)
        elif not eyes_closed_now and not mouth_open_now:
            self.fatigue_score = max(0.0, self.fatigue_score - 0.4)

        state_changed = state != self._last_logged_state
        if state_changed and self.logger is not None:
            self.logger.log_event(state, ear, mar, self.ema_pitch, self.ema_yaw)
            self._last_logged_state = state

        fatigue_confidence = int(min(100, (self.fatigue_score / 30.0) * 100))
        metrics.update({
            "ear": round(float(ear), 2), "mar": round(float(mar), 2),
            "pitch": int(self.ema_pitch), "yaw": int(self.ema_yaw),
            "fatigue_score": round(self.fatigue_score, 1),
            "fatigue_confidence": fatigue_confidence,
            "blink_count": self.blink_count, "yawn_count": self.yawn_count,
            "is_yawning": self.is_yawning,
            "alert": state, "state": state,
            "audio_alert": state in {"DROWSY", "DANGER"},
            "audio_src": ALERT_AUDIO_SRC,
            "state_changed": state_changed,
            "fps": round(self.processing_fps, 1),
        })

        # --- HUD overlay ---
        cv2.polylines(frame, [np.array(left_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        cv2.polylines(frame, [np.array(right_eye, dtype=np.int32)], True, (0, 255, 255), 1)
        mouth_color = (0, 165, 255) if self.is_yawning else (0, 255, 0)
        cv2.polylines(frame, [np.array(mouth, dtype=np.int32)], True, mouth_color, 2)

        cv2.putText(frame, f"Fatigue: {metrics['fatigue_score']} ({fatigue_confidence}%)",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"EAR: {metrics['ear']} | MAR: {metrics['mar']}",
                    (20, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
        cv2.putText(frame, f"Head (P/Y): {metrics['pitch']}/{metrics['yaw']}",
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 2)
        cv2.putText(frame, f"FPS: {metrics['fps']}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if state != "NORMAL":
            color = (0, 0, 255) if state == "DANGER" else (0, 255, 255)
            cv2.putText(frame, f"ALERT: {state}", (20, 145),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)

        return frame, metrics