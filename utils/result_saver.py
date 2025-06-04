import os
import pandas as pd

def flatten_result(result_data):
    return {
        "timestamp": result_data["timestamp"],

        "left_slope_diff": result_data["Left"]["slope_diff"],
        "left_y_diff_0": result_data["Left"]["y_diffs"][0],
        "left_y_diff_1": result_data["Left"]["y_diffs"][1],
        "left_y_diff_2": result_data["Left"]["y_diffs"][2],
        "left_y_diff_3": result_data["Left"]["y_diffs"][3],
        "left_y_diff_4": result_data["Left"]["y_diffs"][4],
        "left_drift_detected": result_data["Left"]["drift_detected"],
        "left_drop_detected": result_data["Left"]["drop_detected"],

        "right_slope_diff": result_data["Right"]["slope_diff"],
        "right_y_diff_0": result_data["Right"]["y_diffs"][0],
        "right_y_diff_1": result_data["Right"]["y_diffs"][1],
        "right_y_diff_2": result_data["Right"]["y_diffs"][2],
        "right_y_diff_3": result_data["Right"]["y_diffs"][3],
        "right_y_diff_4": result_data["Right"]["y_diffs"][4],
        "right_drift_detected": result_data["Right"]["drift_detected"],
        "right_drop_detected": result_data["Right"]["drop_detected"],

        "final_diagnosis": result_data["final_diagnosis"]
    }

def save_result_csv(result_data, save_dir="data", filename="results.csv"):
    """
    결과를 지정한 CSV 파일로 저장하는 함수
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    flattened = flatten_result(result_data)
    df = pd.DataFrame([flattened])
    df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))
