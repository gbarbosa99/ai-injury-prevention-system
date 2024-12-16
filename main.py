import cv2
from src.pose_detection import initialize_pose_detector, process_frame, draw_landmarks
from src.feedback_generator import generate_feedback
from src.risk_analysis import identify_risk_zones, aggregate_risk_feedback

def main():
    """
    Main function to execute the AI Workout analysis pipeline.
    """
    # Initialize pose detector and drawing utilities
    pose, drawing_utils = initialize_pose_detector()

    # Define risk thresholds (example for squat analysis)
    thresholds = {
        "knee_min": 70,  # Minimum safe knee angle in degrees
        "knee_max": 120  # Maximum safe knee angle in degrees
    }

    # Open video capture (webcam or video file)
    cap = cv2.VideoCapture(0)  # Use 0 for webcam; replace with file path for videos

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Unable to access the camera or video file.")
            break

        # Process the frame for pose landmarks
        results = process_frame(pose, frame)

        # Draw landmarks on the frame
        frame_with_landmarks = draw_landmarks(frame, results, pose, drawing_utils)

        # Generate feedback if pose landmarks are detected
        if results.pose_landmarks:
            # Generate feedback
            feedback = generate_feedback(results)

            # Analyze risks
            risks = identify_risk_zones(results.pose_landmarks.landmark, thresholds)
            risk_feedback = aggregate_risk_feedback(risks)

            # Overlay feedback on the frame
            y_offset = 30
            for text in feedback + [risk_feedback]:
                cv2.putText(
                    frame_with_landmarks,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    1,
                    cv2.LINE_AA
                )
                y_offset += 20

        # Display the annotated frame
        cv2.imshow('AI Workout Analysis', frame_with_landmarks)

        # Exit on 'q' key press
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()