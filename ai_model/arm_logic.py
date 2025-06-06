def is_pronator_drift_by_slope(thumb_tip_start, pinky_tip_start, thumb_tip_end, pinky_tip_end, threshold=0.2):
    """
    손목 회전 여부를 판단하는 함수 (엄지-TIP4, 소지-TIP20 기준)
    - slope = (y2 - y1) / (x2 - x1)
    - slope 변화량이 threshold를 넘으면 회전으로 판단
    """
    def compute_slope(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        return dy / dx if abs(dx) > 1e-6 else float('inf')

    slope_start = compute_slope(thumb_tip_start, pinky_tip_start)
    slope_end = compute_slope(thumb_tip_end,   pinky_tip_end)
    slope_diff = abs(slope_end - slope_start)

    drifted = slope_diff > threshold
    return drifted, slope_diff

def is_arm_dropped(y_first, y_last, threshold=0.05):
    """
    각 손가락의 y 좌표 변화량이 threshold를 모두 초과하면 '하강'으로 판단
    """
    diffs = [y2 - y1 for y1, y2 in zip(y_first, y_last)]
    dropped = all(diff > threshold for diff in diffs)
    return dropped, diffs
