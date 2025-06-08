import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1. 모델 로드
model = joblib.load('arm_model.pkl')

# 2. 테스트 데이터 로드 및 전처리
df = pd.read_csv('../data/test_data.csv')  # 테스트 파일 경로

# 'both_abnormal' 제거
df = df[df['final_diagnosis'] != 'both_abnormal']

# 라벨링
df['label'] = df['final_diagnosis'].apply(lambda x: 0 if x == 'normal' else 1)

# feature columns
feature_cols = [
    'left_start_slope', 'left_end_slope', 'left_slope_diff',
    'right_start_slope', 'right_end_slope', 'right_slope_diff',
    'left_y0', 'left_y1', 'left_y2', 'left_y3', 'left_y4',
    'right_y0', 'right_y1', 'right_y2', 'right_y3', 'right_y4',
]
X_test = df[feature_cols].fillna(0)
y_true = df['label']

# 3. 예측 (가중치 수정도 여기서)
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.23).astype(int)

# 4. 결과 저장 및 출력
df['predict_prob'] = probs
df['predicted_label'] = preds
df['predicted_diagnosis'] = df['predicted_label'].apply(lambda x: '비정상' if x == 1 else '정상')

# 5. 성능 평가
acc = accuracy_score(y_true, preds)
rec = recall_score(y_true, preds)
prec = precision_score(y_true, preds)
f1 = f1_score(y_true, preds)

print("\n📊 === 테스트 데이터 평가 결과 ===")
print(f"Accuracy:  {acc:.3f}")
print(f"Recall:    {rec:.3f}")
print(f"Precision: {prec:.3f}")
print(f"F1 Score:  {f1:.3f}")

# 6. 결과 미리보기
print("\n[예측 결과 미리보기]")
print(df[['final_diagnosis', 'predict_prob', 'predicted_diagnosis']].head())

# 7. 저장
df.to_csv('../data/test_result.csv', index=False)
print("\n✅ 예측 결과가 test_result.csv로 저장되었습니다.")
