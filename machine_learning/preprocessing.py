import pandas as pd
import ast

def preprocess_results_csv(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    processed_rows = []

    for _, row in df.iterrows():
        left_dict = safe_eval(row.get("Left", ""))
        right_dict = safe_eval(row.get("Right", ""))

        if not isinstance(left_dict, dict) or not isinstance(right_dict, dict):
            continue
        if "thumb_diff" not in left_dict or "thumb_diff" not in right_dict:
            continue

        left_y_diffs = left_dict.get("y_diffs", [0]*5)
        right_y_diffs = right_dict.get("y_diffs", [0]*5)

        # 손가락 이름에 맞춰서 각 y 변화량 저장
        features = {
            "left_thumb_diff": left_dict.get("thumb_diff", 0),
            "left_pinky_diff": left_dict.get("pinky_diff", 0),
            "left_y_thumb": left_y_diffs[0],
            "left_y_index": left_y_diffs[1],
            "left_y_middle": left_y_diffs[2],
            "left_y_ring": left_y_diffs[3],
            "left_y_pinky": left_y_diffs[4],

            "right_thumb_diff": right_dict.get("thumb_diff", 0),
            "right_pinky_diff": right_dict.get("pinky_diff", 0),
            "right_y_thumb": right_y_diffs[0],
            "right_y_index": right_y_diffs[1],
            "right_y_middle": right_y_diffs[2],
            "right_y_ring": right_y_diffs[3],
            "right_y_pinky": right_y_diffs[4],

            # "drift_detected": int(left_dict.get("drift_detected", False) or right_dict.get("drift_detected", False)),
            # "drop_detected": int(left_dict.get("drop_detected", False) or right_dict.get("drop_detected", False)),

            "label": 0 if str(row.get("final_diagnosis", "")).strip().lower() == "normal" else 1
        }

        processed_rows.append(features)

    return pd.DataFrame(processed_rows)

def safe_eval(cell: str):
    try:
        if isinstance(cell, str) and cell.startswith('{') and cell.endswith('}'):
            return ast.literal_eval(cell)
        else:
            return {}
    except Exception:
        return {}
