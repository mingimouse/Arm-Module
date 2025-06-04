import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score
)
import joblib
import os
from xgboost import XGBClassifier
from preprocessing import preprocess_results_csv

# 1. 데이터 전처리
csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/results.csv"))
print("[INFO] CSV 불러오는 중...")
df = preprocess_results_csv(csv_path)
print(f"[INFO] 총 학습 데이터 수: {len(df)}개")

# 2. X (입력), y (라벨) 나누기
X = df.drop("label", axis=1)
y = df["label"]

# 3. 학습/검증 분할
if len(df) > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
else:
    X_train, X_test, y_train, y_test = X, X, y, y

# 4. 모델 학습
print("[INFO] XGBoost 모델 학습 중...")
model = XGBClassifier(
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='logloss',
    scale_pos_weight=2.0,
    random_state=42
)
model.fit(X_train, y_train)

# 5. 평가
print("\n[RESULT] 테스트셋 평가 결과:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. 추가 지표 출력
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
pr_auc = average_precision_score(y_test, model.predict_proba(X_test)[:, 1])
report = classification_report(y_test, y_pred, output_dict=True)
recall_gap = abs(report['0']['recall'] - report['1']['recall'])

print(f"\n📊 Precision:     {precision:.2f}")
print(f"📊 Recall:        {recall:.2f}")
print(f"📊 F1 Score:      {f1:.2f}")
print(f"📊 PR-AUC:        {pr_auc:.2f}")
print(f"📊 Recall Gap:    {recall_gap:.2f}")

# 7. 모델 저장
save_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "model", "arm_rf_model.pkl"))
os.makedirs(os.path.dirname(save_path), exist_ok=True)
joblib.dump(model, save_path)
print(f"\n✅ 모델이 '{save_path}' 로 저장되었습니다.")
