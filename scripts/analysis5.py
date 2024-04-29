from sklearn.model_selection import cross_val_score

import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 데이터 준비
X = data.iloc[:, :-1]  # 특성
y = data['species']    # 타겟

# 훈련 및 테스트 세트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성 및 훈련
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 예측 및 평가
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# 교차 검증 수행
scores = cross_val_score(model, X, y, cv=5)  # 5-폴드 교차 검증
print("Cross-validated scores:", scores)
print("Average score:", scores.mean())