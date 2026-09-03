# SmartVision — Real-Time Driver Fatigue Monitoring

SmartVision watches a driver through a webcam and flags signs of fatigue in real time — drowsy eyes, yawning, and head-nodding — before they turn into a real problem on the road. It runs entirely on-device (no cloud calls, no data leaving the machine), streams the annotated video feed and live telemetry to a browser dashboard over WebSockets, and sounds an audible alert the moment things escalate.

This isn't a research demo that just draws boxes on a face — it's built around a duration-based state machine, so it actually tries to tell the difference between "you blinked" and "you fell asleep."

---

## What it actually does

- **Tracks your face in real time** using MediaPipe's FaceLandmarker (468-point mesh), running in `VIDEO` mode so it tracks between frames instead of re-detecting from scratch every time.
- **Computes three signals per frame:**
  - **EAR** (Eye Aspect Ratio) — how open your eyes are
  - **MAR** (Mouth Aspect Ratio) — whether you're yawning
  - **Head pose** (pitch/yaw) — whether your head is drooping or turned away, via `solvePnP`
- **Smooths all three** with exponential moving averages, so a single noisy frame (camera jitter, a fast blink) can't flip your state.
- **Escalates through three states** — `NORMAL → DROWSY → DANGER` — based on how *long* a condition persists, not how many frames it lasted, so behavior stays consistent regardless of how fast or slow the machine processing it is.
- **Plays an audio alert** (`assets/alert.wav`) on both the server and the browser when you hit `DROWSY`/`DANGER`, with a cooldown so it doesn't stutter or double-fire.
- **Logs every state transition** to `logs/fatigue_log.csv` in the background, without blocking the video pipeline.
- **Streams live** to a dashboard at `http://localhost:8000` — annotated video feed, EAR/MAR readouts, head angle, blink/yawn counters, a fatigue-risk meter, and processing FPS.

---

## Project structure

```
SmartVision/
├── assets/
│   └── alert.wav                          # audio alert played on DROWSY/DANGER
├── dataset/                                # training data (if you're retraining eye_cnn.h5)
├── logs/
│   └── fatigue_log.csv                     # auto-generated event log (created on first run)
├── models/
│   ├── face_landmarker.task                # MediaPipe model — this is what's actually in use
│   ├── eye_cnn.h5                          # legacy/experimental — not wired into the live pipeline
│   └── shape_predictor_68_face_landmarks.dat  # legacy dlib model — not wired into the live pipeline
├── src/
│   ├── main.py                             # FastAPI app, camera loop, audio, websockets
│   ├── train_cnn.py                        # training script for eye_cnn.h5
│   ├── utils.py
│   └── modules/
│       ├── detector.py                     # the state machine — the core of the app
│       ├── metrics.py                      # EAR / MAR math
│       ├── pose.py                         # head pose estimation (solvePnP)
│       └── logger.py                       # non-blocking CSV event logger
├── templates/
│   └── index.html                          # the dashboard UI
├── tests/
├── requirements.txt
└── README.md
```

> **Heads up:** `eye_cnn.h5` and `shape_predictor_68_face_landmarks.dat` are present in `models/` but the live detection pipeline runs entirely on MediaPipe's `face_landmarker.task`. If you're not actively using the CNN/dlib path, it's worth pruning those files or clearly marking them as experimental so future-you (or a collaborator) doesn't assume they're load-bearing.

---

## Getting started

### 1. Prerequisites
- Python 3.10+ (a `smartvision310/` virtual environment is already set up in this repo)
- A working webcam
- macOS, Windows, or Linux (audio alerts work on all three)

### 2. Set up the environment

```bash
# from the project root
python -m venv smartvision310        # skip if it already exists
source smartvision310/bin/activate   # Windows: smartvision310\Scripts\activate

pip install -r requirements.txt
```

### 3. Run it

```bash
# from the project root — the -m flag matters, see note below
python -m src.main
```

You'll see staged startup logs as the model loads and the camera opens:

```
[Startup] Loading face landmarker model (this can take a few seconds)...
[Startup] Model loaded and warmed up.
[Startup] Server is ready. Open http://localhost:8000
[Camera] Opened successfully.
```

Then open **http://localhost:8000** in your browser, click **Enable Audio System** once (browsers block autoplay until you interact with the page), and you're live.

> **Why `python -m src.main` and not `python src/main.py`?**
> `main.py` imports its own modules as `from src.modules.detector import ...`, which is a package-relative import. Running the file directly changes how Python resolves that import and it'll throw a `ModuleNotFoundError`. Running it as a module (`-m`) from the project root keeps the package structure intact.

---

## How the fatigue detection actually works

The heart of the app is `src/modules/detector.py`. Here's the logic, plainly:

1. **Every frame**, MediaPipe finds your face and hands back landmark coordinates.
2. From those landmarks we compute raw EAR, MAR, and pitch/yaw, then **smooth each one** with an exponential moving average (`α ≈ 0.3–0.4`) so single noisy frames don't matter.
3. We track **how long, continuously**, each condition has been true — eyes closed, mouth open past the yawn threshold, head tilted past the pose threshold. Not frame counts — actual elapsed seconds, measured with `time.monotonic()`.
4. Those durations are compared against thresholds to decide the state:

| Condition | DROWSY threshold | DANGER threshold |
|---|---|---|
| Eyes closed (EAR < 0.21) | 1.0s | 2.5s |
| Yawning (MAR > 0.6) | 1.2s | 3.0s |
| Head tilt (pitch > 15° / yaw > 22°) | 1.2s | 2.5s (25°/35°) |

Any single condition crossing its threshold is enough to escalate — they don't need to happen together. A driver whose head is drooped for 2.5 seconds straight hits `DANGER` even with their eyes technically "open."

5. On a **state change**, the event is logged and, if the new state is `DROWSY` or `DANGER`, an alert sound fires (throttled to at most once every 2.5 seconds).

These thresholds live as plain class attributes at the top of `FatigueDetector.__init__` — tune them there if you find them too sensitive or not sensitive enough for your setup.

---

## The dashboard

`templates/index.html` connects to two WebSocket endpoints:

- `/ws/stream` — the annotated JPEG video frames
- `/ws/metrics` — a JSON payload of everything (`ear`, `mar`, `pitch`, `yaw`, `state`, `fps`, counters, etc.), pushed ~20 times a second

It shows the live camera feed with eye/mouth overlays drawn on it, a fatigue-risk meter, EAR/MAR readouts, head angle, blink/yawn counters, and processing FPS. The alert badge and border flash when you cross into `DROWSY` or `DANGER`.

---

## Configuration reference

| Setting | Where | Default | What it does |
|---|---|---|---|
| `EYE_AR_THRESH` | `detector.py` | `0.21` | EAR below this counts as "eyes closed" |
| `MAR_YAWN_THRESH` | `detector.py` | `0.6` | MAR above this counts as "yawning" |
| `PITCH_DROWSY_DEG` / `YAW_DROWSY_DEG` | `detector.py` | `15° / 22°` | Head angle considered "off-normal" |
| `AUDIO_COOLDOWN_SEC` | `main.py` | `2.5` | Minimum seconds between alert sounds |
| `INFER_WIDTH` | `detector.py` | `320` | Frame width used for MediaPipe inference (smaller = faster, less accurate at extreme angles) |
| Camera index | `main.py`, `_open_camera()` | `0` | Which webcam to use — hardcoded, change if you have multiple |

---

## Troubleshooting

**Video feed won't connect / page hangs on "Connecting..."**
Check your terminal output first — the startup logs will show exactly where things stopped (model loading, camera opening, etc.). The most common causes are the app crashing before it binds the port (check for a traceback) or the camera failing to open (see below).

**Camera-related errors**
You'll see a clear `[Camera] ERROR` message in the terminal if `cv2.VideoCapture(0)` fails to open. Common causes: another app (Zoom, Teams, etc.) is holding the webcam, or your OS hasn't granted this process camera permission (macOS: *System Settings → Privacy & Security → Camera*; Windows: *Settings → Privacy → Camera*).

**No sound on alerts**
- Make sure you clicked **Enable Audio System** at least once — browsers won't autoplay audio without a user gesture.
- On Linux, server-side playback needs `paplay` or `aplay` on your `PATH` (ships with PulseAudio/ALSA on most desktop distros).

**Low FPS / laggy feed**
Check the FPS readout in the dashboard footer. If it's low, try lowering `INFER_WIDTH` in `detector.py` (e.g. `240`), or reduce the camera capture resolution in `main.py`'s `_open_camera()`.

---

## Known limitations

Being upfront about what this is and isn't:

- **Single-session by design.** State (`latest_frame_bytes`, `latest_metrics`) is global, so it's built for one camera feeding one dashboard, not multiple concurrent viewers or camera sources.
- **No authentication.** Fine on `localhost`; don't expose this directly to the open internet without adding some.
- **No automated tests wired up yet** for the detection logic — the `tests/` folder exists but isn't part of this review.
- **Threshold values are reasonable defaults, not calibrated against real driver data.** Tune them for your setup, lighting, and camera placement.

---

## Roadmap ideas

- Per-session/per-client state instead of global singletons, to support multiple viewers
- Configurable camera index / resolution via a config file or env vars
- Continuous (not just transition) telemetry logging, if you want data for later analysis
- Swap the hardcoded thresholds for a small calibration step (e.g. "look normal for 5 seconds" to set a personal EAR baseline)

---

## Running it in Docker

There's a `Dockerfile` and `docker-compose.yml` in the repo — but read this before reaching for them, because there's a real hardware limitation worth knowing about upfront.

**The honest version:** Docker Desktop on macOS and Windows runs containers inside a lightweight VM with no clean path to your host's USB webcam. That's a platform limitation, not something a Dockerfile can work around. So:

- **Live webcam mode is a native-run feature.** Use `python -m src.main` on your own machine for that.
- **Docker is for the deployable app, and for testing the full pipeline against a video file** instead of a live camera — genuinely useful for verifying the container works without needing camera passthrough.

To run it against a sample video:

```bash
# put a test clip at ./sample_videos/demo.mp4, then:
docker compose up --build
```

`SMARTVISION_CAMERA_SOURCE` in `docker-compose.yml` controls the source — set it to a webcam index (`"0"`) or a file path. When it's a file and playback reaches the end, the app's existing reconnect logic re-opens the same path, which conveniently just loops the demo video.

If you're deploying to an actual Linux host with a physical camera attached, webcam passthrough does work there — uncomment the `devices:` line in `docker-compose.yml` (or pass `--device=/dev/video0` to `docker run`).

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

