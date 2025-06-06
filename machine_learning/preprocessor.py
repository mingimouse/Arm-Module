import pandas as pd
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(csv_path='../data/result.csv', test_size=0.2, random_state=42):
    # 1. 데이터 불러오기
    df = pd.read_csv(csv_path)

    # 2. 사용할 feature 정의
    feature_cols = [
        'left_start_slope', 'left_end_slope', 'left_slope_diff',
        'right_start_slope', 'right_end_slope', 'right_slope_diff',
        'left_y0', 'left_y1', 'left_y2', 'left_y3', 'left_y4',
        'right_y0', 'right_y1', 'right_y2', 'right_y3', 'right_y4',
    ]

    # 3. "both_abnormal" 제외
    df = df[df['final_diagnosis'] != 'both_abnormal']

    # 4. 이진 라벨 생성
    df['label'] = df['final_diagnosis'].apply(lambda x: 0 if x == 'normal' else 1)

    # 5. feature, label 분리
    X = df[feature_cols].fillna(0)
    y = df['label']

    # 6. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    return X_train, X_test, y_train, y_test
