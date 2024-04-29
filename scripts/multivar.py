import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 데이터 불러오기
data = pd.read_csv('data/iris.csv')

# 컬럼 이름 변경
data.columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

from statsmodels.multivariate.manova import MANOVA

# 데이터 준비
X = data.iloc[:, :-1]
y = data['species']

# MANOVA 모델 적용
maov = MANOVA.from_formula('sepal_length + sepal_width + petal_length + petal_width ~ species', data=data)
print(maov.mv_test())
