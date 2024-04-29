import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# 상관관계 행렬 생성
correlation_matrix = data.corr()
print(correlation_matrix)

# 상관관계 행렬의 히트맵 시각화
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Matrix for Iris Data')
plt.show()