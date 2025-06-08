import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib

from machine_learning.preprocessor import load_and_preprocess_data

# 1. 데이터 로드
X_train, X_test, y_train, y_test = load_and_preprocess_data()

# 2. 모델 정의 및 학습
model = XGBClassifier(
    eval_metric='logloss',
    max_depth=3,
    learning_rate=0.1,
    n_estimators=100,
    random_state=42
)
model.fit(X_train, y_train)

# 3. 확률 예측값 생성
probs = model.predict_proba(X_test)[:, 1]  # 클래스 1의 확률

# 4. ROC AUC & PR AUC
roc_auc = roc_auc_score(y_test, probs)
pr_auc = average_precision_score(y_test, probs)

# 5. 교차검증 recall
cv_recall = cross_val_score(model, X_train, y_train, scoring='recall', cv=5).mean()

# 6. 다양한 threshold에 대한 평가
thresholds = np.arange(0.30, 0.61, 0.01)
results = []

for thresh in thresholds:
    preds = (probs >= thresh).astype(int)
    recall = recall_score(y_test, preds)
    precision = precision_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    acc = accuracy_score(y_test, preds)
    gap = recall - cv_recall

    results.append({
        'threshold': round(thresh, 3),
        'recall': round(recall, 3),
        'cv_recall': round(cv_recall, 3),
        'recall_gap': round(gap, 3),
        'precision': round(precision, 3),
        'f1_score': round(f1, 3),
        'accuracy': round(acc, 3),
        'roc_auc': round(roc_auc, 3),
        'pr_auc': round(pr_auc, 3)
    })

# 7. 평가 결과 정리 및 출력
print("\n📊 === 성능 평가 결과 (상위 5개 threshold 기준) ===")
result_df = pd.DataFrame(results)
result_df = result_df.sort_values(by='accuracy', ascending=False).reset_index(drop=True)

# 상위 5개만 보기 좋게 출력
print(result_df[['threshold', 'recall', 'precision', 'f1_score', 'accuracy']].head(5).to_string(index=False))

# 데이터 분할 비율 출력
print(f"\n🧪 데이터 분할 비율: Train={len(y_train)}개 / Test={len(y_test)}개 "
      f"({len(y_train) / (len(y_train) + len(y_test)):.0%} / {len(y_test) / (len(y_train) + len(y_test)):.0%})")

# 8. 모델 저장
joblib.dump(model, 'arm_model.pkl')
print("\n✅ 모델이 현재 디렉터리에 'arm_model.pkl'로 저장되었습니다.")

