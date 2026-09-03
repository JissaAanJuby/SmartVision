from scipy.spatial import distance

def calculate_ear(eye_landmarks: list) -> float:
    """Calculates Eye Aspect Ratio (EAR) given 6 eye landmarks."""
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    return (A + B) / (2.0 * C)

def calculate_mar(mouth_landmarks: list) -> float:
    """Calculates Mouth Aspect Ratio (MAR) given 4 mouth landmarks."""
    vertical = distance.euclidean(mouth_landmarks[0], mouth_landmarks[1])
    horizontal = distance.euclidean(mouth_landmarks[2], mouth_landmarks[3])
    return vertical / horizontal if horizontal > 0 else 0.0