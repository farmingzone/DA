import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')  # 데이터의 실제 경로

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

print(data.head())  # 데이터의 첫 5행을 출력

# 기본 통계 출력
print(data.describe())

# 히스토그램으로 데이터 분포 보기
data['sepal_length'].hist(bins=20)
plt.title('Sepal Length Distribution')
plt.xlabel('Sepal Length')
plt.ylabel('Frequency')
plt.show()
