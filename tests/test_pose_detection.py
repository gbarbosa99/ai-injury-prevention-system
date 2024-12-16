import unittest
import cv2
from src.pose_detection import initialize_pose_detector, process_frame, draw_landmarks

class TestPoseDetection(unittest.TestCase):
    def SetUp(self):
        try:
            self.pose, self.drawing = initialize_pose_detector()
        except Exception as e:
            raise RuntimeError(f"Failed to initialize pose detector: {e}")
        
        self.test_frame = cv2.imread("tests/test_image.jpg")
        if self.test_frame is None:
            raise FileNotFoundError("Test image not found. Ensure 'tests/test_image.jpg' exists.")

    def test_initialize_pose_detector(self):
        """
        Test if the pose detector initializes properly.
        """
        self.assertIsNotNone(self.pose, "Pose detector initialization failed.")
        self.assertIsNotNone(self.drawing, "Drawing utilities initialization failed.")
    
    def test_process_frame(self):
        """
        Test if process_frame detects poses correctly
        """
        results = process_frame(self.pose, self.test_frame)
        self.assertIsNotNone(results, "Pose processing returned None.")
        self.assertTrue(
            hasattr(results, "pose_landmarks"),
            "Pose processing doe not return pose landmarks,",
        )

    def test_draw_landmark(self):
        """
        Test if landmarks are being drawn on the frame.
        """
        results = process_frame(self.pose, self.test_frame)
        annotated_frame = draw_landmarks(self.test_frame, results, self.pose, self.drawing)
        self.assertEqual(annotated_frame.shape, self.test_frame.shape, "output frame dimensions are incorrect.")
        # Saving the frame for visual verification
        cv2.imwrite("tests/output_annotated_frame.jpg", annotated_frame)

    def tearDown(self):
        self.pose.close()

if __name__ == "__main__":
    unittest.main()