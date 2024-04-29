import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 데이터 준비
X = data.iloc[:, :-1]  # 마지막 열(species) 제외

# KMeans 모델 적용
kmeans = KMeans(n_clusters=3)  # Iris 데이터에는 3개의 종이 있기 때문에 클러스터의 수를 3으로 설정
clusters = kmeans.fit_predict(X)

# 클러스터링 결과 시각화
plt.scatter(X['petal_length'], X['petal_width'], c=clusters, cmap='viridis')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Clustering of Iris Data')
plt.show()
