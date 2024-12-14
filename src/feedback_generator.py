import math

def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.

    :param a: Tuple (x, y) representing the first point.
    :param b: Tuple (x, y) representing the second point (vertex).
    :param c: Tuple (x, y) representing the third point.
    :return: Angle in degrees.
    """
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return angle + 360 if angle < 0 else angle

def assess_knee_angle(landmarks, left=True):
    """
    Assesses the knee angle and provides feedback if it is outside the desired range.

    :param landmarks: List of pose landmarks.
    :param left: Boolean indicating whether to analyze the left knee (True) or right knee (False).
    :return: Feedback string indicating whether the knee angle is appropriate.
    """
    # Define the landmarks for the hip, knee, and ankle
    if left:
        hip = landmarks[11]  # LEFT_HIP
        knee = landmarks[13]  # LEFT_KNEE
        ankle = landmarks[15]  # LEFT_ANKLE
    else:
        hip = landmarks[12]  # RIGHT_HIP
        knee = landmarks[14]  # RIGHT_KNEE
        ankle = landmarks[16]  # RIGHT_ANKLE

    # Extract (x, y) coordinates
    hip_coords = (hip.x, hip.y)
    knee_coords = (knee.x, knee.y)
    ankle_coords = (ankle.x, ankle.y)

    # Calculate the knee angle
    knee_angle = calculate_angle(hip_coords, knee_coords, ankle_coords)

    # Provide feedback based on the angle
    if knee_angle > 120:
        return "Warning: Your squat depth is too shallow! Bend your knees more."
    elif knee_angle < 70:
        return "Warning: Your squat depth is too low! Avoid over-bending."
    else:
        return "Good form! Keep it up."

def generate_feedback(results):
    """
    Generates feedback for detected pose landmarks.

    :param results: MediaPipe Pose results containing pose landmarks.
    :return: List of feedback strings.
    """
    if not results.pose_landmarks:
        return ["No pose detected. Ensure your full body is visible in the frame."]

    feedback = []

    # Assess the left and right knee angles
    feedback.append(assess_knee_angle(results.pose_landmarks.landmark, left=True))
    feedback.append(assess_knee_angle(results.pose_landmarks.landmark, left=False))

    return feedback
