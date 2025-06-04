from fastapi import APIRouter
from pydantic import BaseModel
from arm.ai_model.drift_logic import is_pronator_drift_thumb_pinky, is_arm_dropped
from typing import List, Dict

router = APIRouter()


class HandData(BaseModel):
    thumb: float
    pinky: float
    y_coords_start: List[float]
    y_coords_end: List[float]


class ArmRequest(BaseModel):
    Left: HandData | None = None
    Right: HandData | None = None


@router.post("/analyze")
def analyze_arm(data: ArmRequest):
    result = {}

    if data.Left:
        left_drift = is_pronator_drift_thumb_pinky("Left", data.Left.thumb, data.Left.pinky, data.Left.thumb,
                                                   data.Left.pinky)
        left_drop, left_diffs = is_arm_dropped(data.Left.y_coords_start, data.Left.y_coords_end)
        result["Left"] = {
            "drift": left_drift,
            "drop": left_drop,
            "y_diffs": left_diffs
        }

    if data.Right:
        right_drift = is_pronator_drift_thumb_pinky("Right", data.Right.thumb, data.Right.pinky, data.Right.thumb,
                                                    data.Right.pinky)
        right_drop, right_diffs = is_arm_dropped(data.Right.y_coords_start, data.Right.y_coords_end)
        result["Right"] = {
            "drift": right_drift,
            "drop": right_drop,
            "y_diffs": right_diffs
        }

    # 최종 판정
    left_abn = result.get("Left", {}).get("drift") or result.get("Left", {}).get("drop")
    right_abn = result.get("Right", {}).get("drift") or result.get("Right", {}).get("drop")

    if left_abn ^ right_abn:
        result["final_diagnosis"] = "drift_detected"
    elif left_abn and right_abn:
        result["final_diagnosis"] = "both_abnormal"
    else:
        result["final_diagnosis"] = "normal"

    return result