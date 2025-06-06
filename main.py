# 패키지 설치
import cv2
import time
import numpy as np
import json
from datetime import datetime
import os
import pandas as pd

# 사용자 정의 패키지 설치
from ai_model.hand_tracker import HandTracker
from ai_model.arm_logic import is_pronator_drift_by_slope, is_arm_dropped
from utils.draw_korean import draw_korean_text
from utils.result_saver import save_result_csv

# 가이드 이미지 불러오기 (알파 채널 포함)
guide = cv2.imread("guide.png", cv2.IMREAD_UNCHANGED)


# 투명 배경 이미지를 영상에 오버레이하는 함수
def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    x, y = pos
    h, w = img_overlay.shape[:2]
    # 이미지가 화면 밖으로 벗어나지 않도록 자르기
    if x + w > img.shape[1] or y + h > img.shape[0]:
        w = min(w, img.shape[1] - x)
        h = min(h, img.shape[0] - y)
        img_overlay = img_overlay[:h, :w]
        alpha_mask = alpha_mask[:h, :w]
    # RGB 채널별로 알파 블랜딩 적용
    for c in range(3):
        img[y:y + h, x:x + w, c] = (
            alpha_mask * img_overlay[:, :, c] +
            (1 - alpha_mask) * img[y:y + h, x:x + w, c]
        )


# Mediapipe 기반 핸드트래커 초기화
tracker = HandTracker()

# 웹캠 캡쳐 설정
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# 측정에 필요한 변수 초기화
start_time = None
measuring_started = False
first_data, last_data = {}, {}
first_y_data, last_y_data = {}, {}
in_guide_start_time = None

print("양손이 정해진 박스에 들어오면 10초간 측정을 시작합니다...")

# 양손 위치를 확인할 기준 박스 영역 정의
left_box = ((150, 400), (450, 650))
right_box = ((850, 400), (1150, 650))


# 손가락 위치가 박스 안에 들어왔는지 확인하는 함수
def is_in_box(point, box):
    x, y = point
    (x1, y1), (x2, y2) = box
    return x1 <= x <= x2 and y1 <= y <= y2


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)  # 좌우반전 (거울모드)
    height, width, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = tracker.process(img_rgb)
    current_time = time.time()

    # 박스 그리기 (빨간색)
    cv2.rectangle(img, left_box[0], left_box[1], (0, 0, 255), 2)
    cv2.rectangle(img, right_box[0], right_box[1], (0, 0, 255), 2)

    # 하단 가이드 이미지 오버레이
    if guide is not None and guide.shape[2] == 4:
        overlay_rgb = guide[:, :, :3]
        overlay_alpha = guide[:, :, 3] / 255.0
        pos_x = (width - guide.shape[1]) // 2
        pos_y = height - guide.shape[0]
        overlay_image_alpha(img, overlay_rgb, (pos_x, pos_y), overlay_alpha)

    # 손이 감지되었을 경우
    if result.multi_hand_landmarks and result.multi_handedness:
        middle_fingers = {"Left": None, "Right": None}
        # 각 손의 중지 TIP 좌표 저장
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[i].classification[0].label
            mid_tip = hand_landmarks.landmark[12]
            cx, cy = int(mid_tip.x * width), int(mid_tip.y * height)
            middle_fingers[hand_label] = (cx, cy)

        # 측정 시작 조건: 양손이 박스 안에 3초간 유지되었을 때
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

        # 측정 중 이라면 시간 표시
        if measuring_started:
            elapsed = current_time - start_time
            elapsed_int = int(elapsed)
            text = f"측정 시간: {elapsed_int} / 10초"
            position = ((width - len(text) * 16) // 2, height - 50)
            img = draw_korean_text(img, text, position)

        # 손가락 좌표 및 데이터를 추출
        for i, hand_landmarks in enumerate(result.multi_hand_landmarks):
            hand_label = result.multi_handedness[i].classification[0].label
            # 엄지-TIP(landmark 4) 좌표
            thumb_lm = hand_landmarks.landmark[4]
            thumb_x, thumb_y = thumb_lm.x, thumb_lm.y
            # 새끼-TIP(landmark 20) 좌표
            pinky_lm = hand_landmarks.landmark[20]
            pinky_x, pinky_y = pinky_lm.x, pinky_lm.y

            y_list = [hand_landmarks.landmark[idx].y for idx in tracker.fingertip_indices]

            # 손가락 TIP 시각화
            for idx in tracker.fingertip_indices:
                lm = hand_landmarks.landmark[idx]
                cx, cy = int(lm.x * width), int(lm.y * height)
                cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

            # 측정 시점별로 좌표 저장 (3초, 10초 기준)
            if measuring_started:
                elapsed = current_time - start_time
                if 2.5 < elapsed < 3.5:
                    # 시작 시점: 엄지TIP, 새끼TIP 저장
                    first_data[hand_label] = {
                        "thumb_tip": (thumb_x, thumb_y),
                        "pinky_tip": (pinky_x, pinky_y)
                    }
                    first_y_data[hand_label] = y_list

                elif 9.5 < elapsed < 10.5:
                    # 종료 시점: 엄지TIP, 새끼TIP 저장
                    last_data[hand_label] = {
                        "thumb_tip": (thumb_x, thumb_y),
                        "pinky_tip": (pinky_x, pinky_y)
                    }
                    last_y_data[hand_label] = y_list

    else:
        # 손 미인식 시 메시지 출력
        if not measuring_started:
            img = draw_korean_text(img, "손 인식 대기 중...", (width // 2 - 150, height - 50))

    if not measuring_started:
        draw_korean_text(img, "양손을 검정 박스 안에 3초간 유지하세요", (width // 2 - 250, 50), font_size=28)

    cv2.imshow("Pronator Drift Detection", img)

    # 'q'를 누르거나 10초 경과 시 종료
    if cv2.waitKey(1) & 0xFF == ord('q') or (measuring_started and (current_time - start_time > 11)):
        break

cap.release()
cv2.destroyAllWindows()

# ---------------- 결과 분석 --------------------
result_data = {"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
left_drift = right_drift = left_fall = right_fall = False

# 왼손 결과 분석
if "Left" in first_data and "Left" in last_data:
    thumb_1 = first_data["Left"]["thumb_tip"]
    pinky_1 = first_data["Left"]["pinky_tip"]
    thumb_2 = last_data["Left"]["thumb_tip"]
    pinky_2 = last_data["Left"]["pinky_tip"]

    # 기울기 기반 드리프트 감지
    left_drift, left_slope_diff = is_pronator_drift_by_slope(thumb_1, pinky_1, thumb_2, pinky_2)
    # 팔 떨어짐 감지
    left_fall, left_diffs = is_arm_dropped(first_y_data["Left"], last_y_data["Left"])

    # 순수 start/end 기울기를 계산해서 저장
    left_start_slope = (pinky_1[1] - thumb_1[1]) / (pinky_1[0] - thumb_1[0] + 1e-6)
    left_end_slope   = (pinky_2[1] - thumb_2[1]) / (pinky_2[0] - thumb_2[0] + 1e-6)

    result_data["Left"] = {
        "start_slope": round(left_start_slope, 4),
        "end_slope":   round(left_end_slope, 4),
        "slope_diff":  round(left_slope_diff, 4),
        "y_diffs":     [round(d, 4) for d in left_diffs],
        "drift_detected": left_drift,
        "drop_detected":  left_fall
    }

# 오른손 결과 분석
if "Right" in first_data and "Right" in last_data:
    thumb_1 = first_data["Right"]["thumb_tip"]
    pinky_1 = first_data["Right"]["pinky_tip"]
    thumb_2 = last_data["Right"]["thumb_tip"]
    pinky_2 = last_data["Right"]["pinky_tip"]

    # 기울기 기반  드리프트 감지
    right_drift, right_slope_diff = is_pronator_drift_by_slope(thumb_1, pinky_1, thumb_2, pinky_2)
    # 팔 떨어짐 감지
    right_fall, right_diffs = is_arm_dropped(first_y_data["Right"], last_y_data["Right"])

    # 순수 start/end 기울기를 계산해서 저장
    right_start_slope = (pinky_1[1] - thumb_1[1]) / (pinky_1[0] - thumb_1[0] + 1e-6)
    right_end_slope   = (pinky_2[1] - thumb_2[1]) / (pinky_2[0] - thumb_2[0] + 1e-6)

    result_data["Right"] = {
        "start_slope": round(right_start_slope, 4),
        "end_slope":   round(right_end_slope, 4),
        "slope_diff":  round(right_slope_diff, 4),
        "y_diffs":     [round(d, 4) for d in right_diffs],
        "drift_detected": right_drift,
        "drop_detected":  right_fall
    }

# 최종 판정
if (left_drift or left_fall) ^ (right_drift or right_fall):
    result_data["final_diagnosis"] = "detected"
elif (left_drift or left_fall) and (right_drift or right_fall):
    result_data["final_diagnosis"] = "both_abnormal"
else:
    result_data["final_diagnosis"] = "normal"

# CSV 생성
save_result_csv(result_data)

# 터미널 출력
print("\n[Drift Result]")
if "Left" in result_data:
    left = result_data["Left"]
    print(f"[Left] 초기 기울기: {left['start_slope']:.4f}, 최종 기울기: {left['end_slope']:.4f} "
          f"--> 기울기 변화량: {left['slope_diff']:.4f} --> 판단결과({left['drift_detected']})")
if "Right" in result_data:
    right = result_data["Right"]
    print(f"[Right] 초기 기울기: {right['start_slope']:.4f}, 최종 기울기: {right['end_slope']:.4f} "
          f"--> 기울기 변화량: {right['slope_diff']:.4f} --> 판단결과({right['drift_detected']})")

print("\n[Drop Result]")
if "Left" in result_data:
    print(f"[Left] y 변화량: {result_data['Left']['y_diffs']} --> 판단결과({result_data['Left']['drop_detected']})")
if "Right" in result_data:
    print(f"[Right] y 변화량: {result_data['Right']['y_diffs']} --> 판단결과({result_data['Right']['drop_detected']})")

print(f"\n[최종 판정]: {result_data['final_diagnosis']}")
