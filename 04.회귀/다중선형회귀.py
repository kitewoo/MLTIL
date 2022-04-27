# 다중회귀 - 보스턴 주택 가격 데이터셋 활용
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn.metrics import mean_squared_error

from sympy import Li

warnings.filterwarnings('ignore')

from sklearn.datasets import load_boston
boston = load_boston() #인스턴스화

df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['PRICE'] = boston.target

print(df.head())

# feature별 산점도와 선형회귀선
sns.regplot(x='CRIM', y='PRICE', data=df)
plt.show()

_, axs = plt.subplots(3,4, figsize=(16,12)) # 그래프 배열 생성
for i, feature in enumerate(df.columns[1:-1]): 
    row, col = i // 4, i % 4 #//는 몫, %는 나머지 로 배열 순서 생성
    sns.regplot(x=feature, y='PRICE', data=df, ax=axs[row][col])

plt.show()


from sklearn.linear_model import LinearRegression
for feature in df.columns[:-1]:
    lr = LinearRegression() #인스턴스화
    X = df[feature].values.reshape(-1,1) # numpy nparray로 변환 & 행이 여러개에 열은 1개가 나와야 하니까 -1,1 *target은 (1,-1)
    lr.fit(X, boston.target)  #학습
    score = lr.score(X, boston.target) #R_squared(결정력) 값
    print(f'{feature}:\t{score:.4f}')

from sklearn.model_selection import train_test_split #데이터셋 분리, ndarray로 저장
X_train, X_test, y_train, y_test = train_test_split(
    boston.data, boston.target, test_size=0.1, random_state=2022
)

lr = LinearRegression()
lr.fit(X_train, y_train) #학습
lr.score(X_train,y_train) #학습된 데이터들의 R-squred(결정력) 값은 계수로 작용한다.

lr.coef_ #계수, Weight
lr.intercept_ #절편, bias

# 회귀식 y = 시그마 ( coef_ * 각 요소)

print(np.dot(lr.coef_, X_test[0]) + lr.intercept_) # dot는 행렬 연산 함수, #결과 : 21.2275  #의미 : 첫번째 테스트셋의 다중회귀 연산한 y값
print(lr.predict(X_test[0].reshape(1,-1))) # ndarray : [[........]] #결과 : 21.2275

# 10개의 테스트에 대해서 적용 (dot와 predict 비교)
for i in range(10):
    pred1 = np.dot(lr.coef_, X_test[i]) + lr.intercept_
    pred2 = lr.predict(X_test[i].reshape(1,-1))
    print(f'실제값: {y_test[i]}, \t직접 계산 예측값: {pred1:.4f}, \tLR 예측값: {pred2[0]:.4f}')


# 회귀도 여러 회귀가 있다. 
# 선형회귀
from sklearn.metrics import r2_score,mean_absolute_error
pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test,pred_lr)
mse_lr = mean_squared_error(y_test, pred_lr)

#Decision Tree regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=2022)
dtr.fit(X_train,y_train)

pred_dt = dtr.predict(X_test)
r2_dt = r2_score(y_test, pred_dt)
mse_dt = mean_squared_error(y_test, pred_dt)

#SVM
from sklearn.svm import SVR
svr =SVR()
svr.fit(X_train,y_train)
pred_sv = svr.predict(X_test)
r2_sv = r2_score(y_test, pred_sv)
mse_sv = mean_squared_error(y_test, pred_sv)

#RandomForest
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=2022)
rfr.fit(X_train, y_train)
pred_rf = rfr.predict(X_test)
r2_rf = r2_score(y_test, pred_rf)
mse_rf = mean_squared_error(y_test, pred_rf)

#XGBoost
from xgboost import XGBRegressor
xgr = XGBRegressor(random_state=2022)
xgr.fit(X_train, y_train)
pred_xg = xgr.predict(X_test)
r2_xg = r2_score(y_test, pred_xg)
mse_xg = mean_squared_error(y_test, pred_xg)

#회귀 방식별 비교
print('LR\tDT\tSVM\tRF\tXG')
print(f'{r2_lr:.4f}\t{r2_dt:.4f}\t{r2_sv:.4f}\t{r2_rf:.4f}\t{r2_xg:.4f}')
print(f'{mse_lr:.4f}\t{mse_dt:.4f}\t{mse_sv:.4f}\t{mse_rf:.4f}\t{mse_xg:.4f}')

df = pd.DataFrame({
    'y_test':y_test, 'LR':pred_lr, 'DT':pred_dt, 'SVM':pred_sv, 'RF':pred_rf, 'XG':pred_xg
})

print(df.head())



