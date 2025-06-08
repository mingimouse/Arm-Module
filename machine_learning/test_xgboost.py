import pandas as pd
import joblib
from xgboost import XGBClassifier

# 1. 모델 로드
model = joblib.load('arm_model.pkl')

# 2. 테스트할 CSV 데이터 불러오기
df = pd.read_csv('../data/test_sample.csv')  # 테스트 CSV 경로
feature_cols = [
    'left_start_slope', 'left_end_slope', 'left_slope_diff',
    'right_start_slope', 'right_end_slope', 'right_slope_diff',
    'left_y0', 'left_y1', 'left_y2', 'left_y3', 'left_y4',
    'right_y0', 'right_y1', 'right_y2', 'right_y3', 'right_y4',
]

X_test = df[feature_cols].fillna(0)

# 3. 예측 확률 및 결과 출력
probs = model.predict_proba(X_test)[:, 1]
preds = (probs >= 0.5).astype(int)  # threshold=0.5 기준

# 4. 출력
df['predict_prob'] = probs
df['predicted_label'] = preds
df['predicted_diagnosis'] = df['predicted_label'].apply(lambda x: '비정상' if x == 1 else '정상')

print(df[['predict_prob', 'predicted_diagnosis']])

# 5. 결과 저장 (선택)
df.to_csv('../data/test_result.csv', index=False)
