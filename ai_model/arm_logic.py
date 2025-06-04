def is_pronator_drift_thumb_pinky(hand_label, mcp5_start_x, mcp13_start_x, mcp5_end_x, mcp13_end_x, threshold=0.05):
    """
    손목이 회전했는지 판단 (검지-MCP와 약지-MCP 사이 기울기 변화 사용)
    - 시작/종료 시점의 mcp5, mcp13 x좌표 비교로 회전 판단
    """
    start_slope = mcp13_start_x - mcp5_start_x
    end_slope = mcp13_end_x - mcp5_end_x
    slope_diff = abs(end_slope - start_slope)

    print(f"[{hand_label}] 기울기 변화량: {round(slope_diff, 4)}")
    return slope_diff > threshold


def is_arm_dropped(y_first, y_last, threshold=0.05):
    """
       각 손가락의 y 좌표 변화량이 threshold를 모두 초과하면 '하강'으로 판단
    """
    diffs = [y2 - y1 for y1, y2 in zip(y_first, y_last)]
    dropped = all(diff > threshold for diff in diffs)
    return dropped, diffs
