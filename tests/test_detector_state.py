import unittest

from src.modules.detector import FatigueDetector


class DetectorStateTests(unittest.TestCase):
    def test_drowsy_requires_persistent_low_ear(self):
        detector = FatigueDetector(model_path="models/face_landmarker.task")

        for _ in range(detector.DROWSY_EYE_FRAMES - 1):
            state = detector.evaluate_alert_state(ear=0.10, mar=0.2, pitch=0, yaw=0)
            self.assertEqual(state, "NORMAL")

        state = detector.evaluate_alert_state(ear=0.10, mar=0.2, pitch=0, yaw=0)
        self.assertEqual(state, "DROWSY")

    def test_danger_requires_persistent_yawn(self):
        detector = FatigueDetector(model_path="models/face_landmarker.task")

        for _ in range(detector.DANGER_YAWN_FRAMES - 1):
            state = detector.evaluate_alert_state(ear=0.24, mar=0.9, pitch=0, yaw=0)
            self.assertNotEqual(state, "DANGER")

        state = detector.evaluate_alert_state(ear=0.24, mar=0.9, pitch=0, yaw=0)
        self.assertEqual(state, "DANGER")


if __name__ == "__main__":
    unittest.main()
