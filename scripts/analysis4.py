import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

from sklearn.decomposition import PCA

# PCA 모델 생성 및 학습
pca = PCA(n_components=2)
principal_components = pca.fit_transform(data.iloc[:, :-1])  # 마지막 열(species) 제외

# 결과 DataFrame 생성
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
pca_df['species'] = data['species']

# PCA 결과 시각화
sns.scatterplot(x='PC1', y='PC2', hue='species', data=pca_df)
plt.title('PCA of Iris Dataset')
plt.show()
