from sklearn.datasets import load_iris
import pandas as pd

# 데이터 로드
iris = load_iris()
iris_df = pd.DataFrame(data= iris.data, columns= iris.feature_names)
iris_df['species'] = iris.target

# CSV 파일로 저장
iris_df.to_csv('iris.csv', index=False)
