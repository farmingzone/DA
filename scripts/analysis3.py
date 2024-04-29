import pandas as pd
import matplotlib.pyplot as plt

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

# 종별로 그룹화하여 기술 통계 출력
grouped_data = data.groupby('species').describe()
print(grouped_data)

# 종별로 sepal length의 평균을 시각화
species_sepal_length_mean = data.groupby('species')['sepal_length'].mean()
species_sepal_length_mean.plot(kind='bar')
plt.title('Average Sepal Length by Species')
plt.ylabel('Sepal Length')
plt.show()
