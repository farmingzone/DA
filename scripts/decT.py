import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']



from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 데이터 준비
X = data.iloc[:, :-1]
y = data['species']

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 모델 생성 및 학습
tree_model = DecisionTreeClassifier()
tree_model.fit(X_train, y_train)

# 예측 및 평가
tree_predictions = tree_model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, tree_predictions))
