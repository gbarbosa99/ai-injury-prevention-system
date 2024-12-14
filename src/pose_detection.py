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
def run_pose_detection(viideo_capture=0):
    """
    Runs the pose detection pipeline on the given video source

    parameters:
    - video_source: path to video gile or integer for webcam
    """
    # Initialize MediaPipe Pose and Drawing modules
    mp_pose = mp.solutions.pose
    pose = mp.pose.Pose()
    mp_drawing = mp.solutions.drawing_utils

    # Open video capture 
    cap = cv2.VideoCapture(video_source)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to access the camera of video file.")
            break

        # Convert the image to RGB because MediaPipe requires RGB input
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame to detect poses
        results = pose.process(rgb_frame)

        # Draw the pose landmarks and connections on the frame
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
        
        # Display the frame with annotations 
        cv2.imshow('Pose Detection', frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.detroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pose Detection Script')
    parser.add_argument(
        "--video_source",
        type=str,
        default=0,
        help="Path to video file or webcam index (default: 0 for webcam)."
    )
    args = parser.parse_args()

    run_pose_detection(args.video_source)