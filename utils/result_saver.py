import os
import pandas as pd

def flatten_result(result):
    flat = {
        "timestamp": result.get("timestamp", "")
    }

    left = result.get("Left", {})
    flat["left_start_slope"] = left.get("start_slope")
    flat["left_end_slope"] = left.get("end_slope")
    flat["left_slope_diff"] = left.get("slope_diff")
    flat["left_drift"] = left.get("drift_detected")
    for i in range(5):
        flat[f"left_y{i}"] = left.get("y_diffs", [None]*5)[i]
    flat["left_drop"] = left.get("drop_detected")

    right = result.get("Right", {})
    flat["right_start_slope"] = right.get("start_slope")
    flat["right_end_slope"] = right.get("end_slope")
    flat["right_slope_diff"] = right.get("slope_diff")
    flat["right_drift"] = right.get("drift_detected")
    for i in range(5):
        flat[f"right_y{i}"] = right.get("y_diffs", [None]*5)[i]
    flat["right_drop"] = right.get("drop_detected")

    flat["final_diagnosis"] = result.get("final_diagnosis", "")
    return flat

def save_result_csv(result_data, save_path="data/result.csv"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    flattened = flatten_result(result_data)
    df = pd.DataFrame([flattened])
    df.to_csv(save_path, mode='a', index=False, header=not os.path.exists(save_path))
