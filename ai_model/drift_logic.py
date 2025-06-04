def is_pronator_drift_thumb_pinky(hand_label, thumb_x1, pinky_x1, thumb_x2, pinky_x2, threshold=0.020):
    thumb_diff = thumb_x2 - thumb_x1
    pinky_diff = pinky_x2 - pinky_x1
    if hand_label == "Right":
        return thumb_diff < -threshold and pinky_diff > threshold
    elif hand_label == "Left":
        return thumb_diff > threshold and pinky_diff < -threshold
    else:
        return False

def is_arm_dropped(y_first, y_last, threshold=0.05):
    diffs = [y2 - y1 for y1, y2 in zip(y_first, y_last)]
    dropped = all(diff > threshold for diff in diffs)
    return dropped, diffs
