import cv2
import time
import numpy as np
import json
from datetime import datetime
import os
import pandas as pd

from ai_model.hand_tracker import HandTracker
from ai_model.drift_logic import is_pronator_drift_thumb_pinky, is_arm_dropped
from utils.draw import draw_korean_text

# 가이드 이미지 불러오기 (알파 채널 포함)
guide = cv2.imread("guide.png", cv2.IMREAD_UNCHANGED)

def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img_overlay.shape[:2]
    if x + w > img.shape[1] or y + h > img.shape[0]:
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        img_overlay = img_overlay[:h, :w]
        alpha_mask = alpha_mask[:h, :w]
    for c in range(3):
        img[y:y+h, x:x+w, c] = (
            alpha_mask * img_overlay[:, :, c] +
            (1 - alpha_mask) * img[y:y+h, x:x+w, c]
        )

tracker = HandTracker()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

start_time = None
measuring_started = False
first_data, last_data = {}, {}
first_y_data, last_y_data = {}, {}
in_guide_start_time = None
print("양손이 정해진 박스에 들어오면 10초간 측정을 시작합니다...")

# 박스 영역 정의
left_box = ((150, 400), (450, 650))
right_box = ((850, 400), (1150, 650))

def is_in_box(point, box):
    x, y = point
    (x1, y1), (x2, y2) = box
    return x1 <= x <= x2 and y1 <= y <= y2

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    height, width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = tracker.process(img_rgb)
    current_time = time.time()

    # 박스 그리기 (검정색)
    cv2.rectangle(img, left_box[0], left_box[1], (0, 0, 255), 2)
    cv2.rectangle(img, right_box[0], right_box[1], (0, 0, 255), 2)

    # 가이드 이미지 오버레이
    if guide is not None and guide.shape[2] == 4:
        overlay_rgb = guide[:, :, :3]
        overlay_alpha = guide[:, :, 3] / 255.0
        pos_x = (width - guide.shape[1]) // 2
        pos_y = height - guide.shape[0]
        overlay_image_alpha(img, overlay_rgb, (pos_x, pos_y), overlay_alpha)

    if result.multi_hand_landmarks and result.multi_handedness:
        middle_fingers = {"Left": None, "Right": None}
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[i].classification[0].label
            mid_tip = hand_landmarks.landmark[12]
            cx, cy = int(mid_tip.x * width), int(mid_tip.y * height)
            middle_fingers[hand_label] = (cx, cy)

        if not measuring_started and middle_fingers["Left"] and middle_fingers["Right"]:
            in_left = is_in_box(middle_fingers["Left"], left_box)
            in_right = is_in_box(middle_fingers["Right"], right_box)

            if in_left and in_right:
                if in_guide_start_time is None:
                    in_guide_start_time = current_time
                elif current_time - in_guide_start_time > 3:
                    start_time = current_time
                    measuring_started = True
                    print("측정 시작!")
            else:
                in_guide_start_time = None

        if measuring_started:
            elapsed = current_time - start_time
            elapsed_int = int(elapsed)
            text = f"측정 시간: {elapsed_int} / 10초"
            position = ((width - len(text) * 16) // 2, height - 50)
            img = draw_korean_text(img, text, position)

        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[i].classification[0].label
            thumb_x = hand_landmarks.landmark[4].x
            pinky_x = hand_landmarks.landmark[20].x
            y_list = [hand_landmarks.landmark[idx].y for idx in tracker.fingertip_indices]
            for idx in tracker.fingertip_indices:
                lm = hand_landmarks.landmark[idx]
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)
            if measuring_started:
                elapsed = current_time - start_time
                if 2.5 < elapsed < 3.5:
                    first_data[hand_label] = (thumb_x, pinky_x)
                    first_y_data[hand_label] = y_list
                elif 9.5 < elapsed < 10.5:
                    last_data[hand_label] = (thumb_x, pinky_x)
                    last_y_data[hand_label] = y_list
    else:
        if not measuring_started:
            img = draw_korean_text(img, "손 인식 대기 중...", (width // 2 - 150, height - 50))

    if not measuring_started:
        draw_korean_text(img, "양손을 검정 박스 안에 3초간 유지하세요", (width // 2 - 250, 50), font_size=28)

    cv2.imshow("Pronator Drift Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or (measuring_started and (current_time - start_time > 11)):
        break

cap.release()
cv2.destroyAllWindows()

# 결과 분석 및 저장
left_drift = right_drift = left_fall = right_fall = False
result_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

if "Left" in first_data and "Left" in last_data:
    t1, p1 = first_data["Left"]
    t2, p2 = last_data["Left"]
    left_drift = is_pronator_drift_thumb_pinky("Left", t1, p1, t2, p2)
    left_fall, left_diffs = is_arm_dropped(first_y_data["Left"], last_y_data["Left"])
    result_data["Left"] = {
        "thumb_diff": round(t2 - t1, 3),
        "pinky_diff": round(p2 - p1, 3),
        "y_diffs": [round(d, 3) for d in left_diffs],
        "drift_detected": left_drift,
        "drop_detected": left_fall
    }

if "Right" in first_data and "Right" in last_data:
    t1, p1 = first_data["Right"]
    t2, p2 = last_data["Right"]
    right_drift = is_pronator_drift_thumb_pinky("Right", t1, p1, t2, p2)
    right_fall, right_diffs = is_arm_dropped(first_y_data["Right"], last_y_data["Right"])
    result_data["Right"] = {
        "thumb_diff": round(t2 - t1, 3),
        "pinky_diff": round(p2 - p1, 3),
        "y_diffs": [round(d, 3) for d in right_diffs],
        "drift_detected": right_drift,
        "drop_detected": right_fall
    }

if (left_drift or left_fall) ^ (right_drift or right_fall):
    result_data["final_diagnosis"] = "drift_detected"
elif (left_drift or left_fall) and (right_drift or right_fall):
    result_data["final_diagnosis"] = "both_abnormal"
else:
    result_data["final_diagnosis"] = "normal"

os.makedirs("data", exist_ok=True)
save_path = "data/results.csv"
df = pd.DataFrame([result_data])
df.to_csv(save_path, mode="a", index=False, header=not os.path.exists(save_path))

print("\n📋 측정 결과 요약:")
if "Left" in result_data:
    left = result_data["Left"]
    print(f"[왼손 thumb 변화량: {left['thumb_diff']}, pinky 변화량: {left['pinky_diff']}]")
    print(f"[왼손 y 변화량: {left['y_diffs']} → 하강: {left['drop_detected']}]")
if "Right" in result_data:
    right = result_data["Right"]
    print(f"[오른손 thumb 변화량: {right['thumb_diff']}, pinky 변화량: {right['pinky_diff']}]")
    print(f"[오른손 y 변화량: {right['y_diffs']} → 하강: {right['drop_detected']}]")
print(f"🔍 최종 판정 결과: {result_data['final_diagnosis']}")
