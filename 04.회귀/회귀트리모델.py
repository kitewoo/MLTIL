'''선형 회귀는 회귀 계수의 관계를 모두 선형으로 가정하는 방식. 
비선형도 선형 함수와 같이 회귀 함수를 구하여 독립 변수를 대입하고 결괏값 예측 회귀계수 결합이 비선형일 뿐

트리모델은 회귀 함수를 기반으로 하지 않고 결정 트리와 같은 트리를 기반으로 하는 회귀방식이다.

작동 방식

1. X값의 균일도나 지니 계수에 따라 분할 후 구간별 평균값을 구해 최종적으로 리프 노드에 평균값을 할당'''

from sklearn.datasets import load_boston
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings("ignore")

boston = load_boston()
bostonDF = pd.DataFrame(boston.data, columns= boston.feature_names)
bostonDF['price'] = boston.target
y_target = bostonDF['price']
x_data = bostonDF.drop(['price'], axis=1, inplace=False)


'''cross val score : 교차 검증으로 K개 씩 그룹을 만들어서 하나는 테스트용 나머지는 훈련용으로 사용하면서 각 분할 당 정확도 도출'''
rf = RandomForestRegressor(random_state=0, n_estimators=1000)
neg_mse_scores = cross_val_score(rf, x_data, y_target, scoring='neg_mean_squared_error', cv=5)
rmse_scores = np.sqrt(-1*neg_mse_scores)
avg_rmse = np.mean(rmse_scores)

print(' 5 cross each negative MSE scores: ', np.round(neg_mse_scores,2))
