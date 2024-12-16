def identify_risk_zones(landmarks, thresholds):
    """
    Identifies body parts or movements that exceed safe thresholds.

    :param landmarks: List of pose landmarks.
    :param thresholds: Dictionary containing joint thresholds (e.g., angles).
    :return: List of risk warnings or identified risks.
    """
    risks = []

    # Example: Check knee angles for squat depth
    left_knee_angle = calculate_angle(
        (landmarks[11].x, landmarks[11].y),  # LEFT_HIP
        (landmarks[13].x, landmarks[13].y),  # LEFT_KNEE
        (landmarks[15].x, landmarks[15].y)   # LEFT_ANKLE
    )

    if left_knee_angle < thresholds['knee_min']:
        risks.append("Left knee over-bending detected.")
    elif left_knee_angle > thresholds['knee_max']:
        risks.append("Left knee insufficient bending detected.")

    # Example: Add similar checks for other joints or thresholds
    right_knee_angle = calculate_angle(
        (landmarks[12].x, landmarks[12].y),  # RIGHT_HIP
        (landmarks[14].x, landmarks[14].y),  # RIGHT_KNEE
        (landmarks[16].x, landmarks[16].y)   # RIGHT_ANKLE
    )

    if right_knee_angle < thresholds['knee_min']:
        risks.append("Right knee over-bending detected.")
    elif right_knee_angle > thresholds['knee_max']:
        risks.append("Right knee insufficient bending detected.")

    return risks

def aggregate_risk_feedback(risks):
    """
    Aggregates identified risks into user-friendly feedback.

    :param risks: List of identified risks.
    :return: Consolidated risk feedback.
    """
    if not risks:
        return "No significant risks detected. Keep up the good form!"

    feedback = "\n".join(risks)
    return f"Risk Analysis:\n{feedback}"

# Helper Function
def calculate_angle(a, b, c):
    """
    Calculates the angle between three points.

    :param a: Tuple (x, y) representing the first point.
    :param b: Tuple (x, y) representing the second point (vertex).
    :param c: Tuple (x, y) representing the third point.
    :return: Angle in degrees.
    """
    import math
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return angle + 360 if angle < 0 else angle
