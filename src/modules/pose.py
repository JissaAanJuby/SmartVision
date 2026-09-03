import cv2
import numpy as np

POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

MODEL_3D_POINTS = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)


def estimate_head_pose(landmarks, img_w: int, img_h: int):
    """Calculates normalized pitch and yaw head angles.

    Returns (pitch, yaw, success). solvePnP's own success flag was
    previously discarded, so a degenerate solve (e.g. from a partially
    occluded or edge-of-frame face) could silently return a garbage
    angle. Callers should hold onto the last known-good angle when
    success is False rather than trusting a bad value.
    """
    image_points = np.array([
        (landmarks[idx].x * img_w, landmarks[idx].y * img_h)
        for idx in POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = img_w
    center = (img_w / 2, img_h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, translation_vec = cv2.solvePnP(
        MODEL_3D_POINTS, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return 0.0, 0.0, False

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

    raw_pitch, yaw = angles[0], angles[1]

    # Normalize pitch angle around 0 degrees
    if raw_pitch > 90:
        pitch = raw_pitch - 180
    elif raw_pitch < -90:
        pitch = raw_pitch + 180
    else:
        pitch = raw_pitch

    return pitch, yaw, True