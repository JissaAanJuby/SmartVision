import os
import sys
import time
import cv2
import asyncio
import subprocess
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.modules.detector import FatigueDetector
from src.modules.logger import FatigueLogger

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "face_landmarker.task"
TEMPLATE_PATH = BASE_DIR / "templates" / "index.html"
ASSETS_DIR = BASE_DIR / "assets"
ALERT_AUDIO_PATH = ASSETS_DIR / "alert.wav"
LOG_PATH = BASE_DIR / "logs" / "fatigue_log.csv"

print(f"[Startup] BASE_DIR = {BASE_DIR}")
print(f"[Startup] Model path exists: {MODEL_PATH.exists()} ({MODEL_PATH})")
print(f"[Startup] Template exists:  {TEMPLATE_PATH.exists()} ({TEMPLATE_PATH})")
print(f"[Startup] Assets dir exists: {ASSETS_DIR.exists()} ({ASSETS_DIR})")

print("[Startup] Initializing CSV logger...")
fatigue_logger = FatigueLogger(LOG_PATH)
print("[Startup] CSV logger ready.")

print("[Startup] Loading face landmarker model (this can take a few seconds)...")
detector = FatigueDetector(model_path=str(MODEL_PATH), logger=fatigue_logger)
print("[Startup] Model loaded and warmed up.")

executor = ThreadPoolExecutor(max_workers=1)

latest_frame_bytes = None
latest_metrics = {
    "ear": 0.0, "mar": 0.0, "pitch": 0, "yaw": 0,
    "blink_count": 0, "yawn_count": 0, "is_yawning": False,
    "fatigue_score": 0.0, "fatigue_confidence": 0, "alert": "NORMAL", "state": "NORMAL",
    "audio_alert": False, "audio_src": "/assets/alert.wav", "fps": 0.0
}

is_processing = False

# --- Audio: single source of truth. detector.py no longer plays sound
# itself -- previously both main.py and detector.py tried to trigger
# alerts independently (different throttle timers, Windows-only in
# both cases), which could double-fire or silently no-op elsewhere. ---
_alert_lock = threading.Lock()
_last_audio_play_at = 0.0
AUDIO_COOLDOWN_SEC = 2.5


def _play_wav_blocking(path: str):
    """Cross-platform WAV playback (previously Windows-only via winsound)."""
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.PlaySound(path, winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NOSTOP)
        elif sys.platform == "darwin":
            subprocess.run(["afplay", path], check=False)
        else:
            for player in ("paplay", "aplay"):
                try:
                    subprocess.run(
                        [player, path], check=True,
                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    return
                except (FileNotFoundError, subprocess.CalledProcessError):
                    continue
            print("[Audio Alert Error]: no usable Linux audio player found (tried paplay, aplay)")
    except Exception as exc:
        print(f"[Audio Alert Error]: {exc}")


def trigger_alert_sound(state: str):
    """Fire-and-forget playback with a cooldown so back-to-back DANGER frames don't stack sounds."""
    global _last_audio_play_at
    if state not in ("DROWSY", "DANGER"):
        return

    now = time.monotonic()
    with _alert_lock:
        if now - _last_audio_play_at < AUDIO_COOLDOWN_SEC:
            return
        _last_audio_play_at = now

    if not ALERT_AUDIO_PATH.exists():
        print(f"[Audio Alert Error]: Missing file at {ALERT_AUDIO_PATH}")
        return

    threading.Thread(target=_play_wav_blocking, args=(str(ALERT_AUDIO_PATH),), daemon=True).start()


def process_worker(frame):
    """Runs off the asyncio event loop thread so heavy inference never blocks the websockets."""
    global latest_frame_bytes, latest_metrics, is_processing
    try:
        annotated_frame, metrics = detector.process_frame(frame)
        if metrics:
            latest_metrics = metrics
            if metrics.get("audio_alert"):
                trigger_alert_sound(metrics.get("state", "NORMAL"))

        _, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 40])
        latest_frame_bytes = buffer.tobytes()
    except Exception as err:
        print(f"[Detector Error]: {err}")
    finally:
        is_processing = False


def _open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print("[Camera] ERROR: cv2.VideoCapture(0) did not open. Check that: "
              "(1) a webcam is connected, (2) no other app is using it, "
              "(3) this process has camera permission (macOS: System Settings > "
              "Privacy & Security > Camera; Windows: Settings > Privacy > Camera).")
    else:
        print("[Camera] Opened successfully.")
    return cap


async def camera_loop():
    """Background camera loop. Retries the camera on repeated read failures instead of
    leaving the stream permanently dead if a USB webcam blips."""
    global is_processing
    cap = _open_camera()
    loop = asyncio.get_running_loop()
    consecutive_failures = 0

    try:
        while True:
            ret, frame = await asyncio.to_thread(cap.read)
            if not ret:
                consecutive_failures += 1
                if consecutive_failures == 1:
                    print("[Camera] Frame read failed, retrying...")
                if consecutive_failures >= 30:
                    print("[Camera] Lost signal for ~0.3s, attempting reconnect...")
                    cap.release()
                    await asyncio.sleep(1.0)
                    cap = _open_camera()
                    consecutive_failures = 0
                await asyncio.sleep(0.01)
                continue

            if consecutive_failures > 0:
                print("[Camera] Frame reads recovered.")
            consecutive_failures = 0

            if not is_processing:
                is_processing = True
                loop.run_in_executor(executor, process_worker, frame.copy())

            await asyncio.sleep(0.01)
    finally:
        cap.release()


app = FastAPI(title="SmartVision API", version="1.0.0")
app.mount("/assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")


@app.on_event("startup")
async def startup_event():
    print("[Startup] FastAPI startup event fired, launching camera loop task...")
    asyncio.create_task(camera_loop())
    print("[Startup] Server is ready. Open http://localhost:8000")


@app.on_event("shutdown")
async def shutdown_event():
    fatigue_logger.stop()


@app.get("/")
async def serve_dashboard():
    return FileResponse(str(TEMPLATE_PATH), media_type="text/html")


@app.websocket("/ws/stream")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("[WS] /ws/stream client connected")
    try:
        while True:
            if latest_frame_bytes is not None:
                await websocket.send_bytes(latest_frame_bytes)
            await asyncio.sleep(0.01)
    except WebSocketDisconnect:
        print("[WS] /ws/stream client disconnected")


@app.websocket("/ws/metrics")
async def metrics_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            await websocket.send_json(latest_metrics)
            await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)