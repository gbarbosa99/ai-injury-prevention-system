import cv2
import mediapipe as mp
import argparse 

"""
Goal: 
Detect human poses in real-time using a webcam and display the results visually. 

Key Features:
- Capture video from the webcam
- Detect body landmarks (joints, elbows, knees)
- Overlay visual markers

Research libraries
- Questions to ask:
   - What tools are available
   - Are there libraries that make this task easier?

OpenCV: popular library for video and image processing
MediaPipe: popular library developed by Google for detecting body landmarks.
"""

def initialize_pose_detector():
    """
    Initializes the MediaPipe Pose and Drawing modules.

    :return: Tuple containing the pose object and drawing utility.
    """
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose()
    mp_drawing = mp.solutions.drawing_utils
    return pose, mp_drawing

def process_frame(pose, frame):
    """
    Processes a video frame to detect pose landmarks.

    :param pose: Initialized MediaPipe Pose object.
    :param frame: Video frame to process.
    :return: Processed frame with pose landmarks drawn and pose results.
    """
    # Convert the image to RGB as MediaPipe requires RGB input
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect poses
    results = pose.process(rgb_frame)

    return results

def draw_landmarks(frame, results, mp_pose, mp_drawing):
    """
    Draws pose landmarks and connections on a video frame.

    :param frame: Video frame to annotate.
    :param results: MediaPipe Pose results.
    :param mp_pose: MediaPipe Pose module.
    :param mp_drawing: MediaPipe Drawing utilities module.
    :return: Annotated frame.
    """
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )
    return frame