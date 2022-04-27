import imp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

#과제 1 : 당뇨병 데이터를 여러 다중회귀 비교

from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

df = pd.DataFrame(diabetes.data, columns = diabetes.feature_names)
df['target'] = diabetes.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, test_size=0.1, random_state=2022)
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)

np.dot(lr.coef_, X_test[0]) + lr.intercept_
lr.predict(X_test[0].reshape(1,-1)) 


#linear regressor
from sklearn.metrics import r2_score, mean_squared_error
pred_lr = lr.predict(X_test)
r2_lr = r2_score(y_test, pred_lr) #
mse_lr = mean_squared_error(y_test, pred_lr) # 

#DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=2022)
dtr.fit(X_train, y_train)
pred_dt = dtr.predict(X_test) 
r2_dt = r2_score(y_test, pred_dt) #
mse_dt = mean_squared_error(y_test, pred_dt) #

#SVM
from sklearn.svm import SVR
svr = SVR()
svr.fit(X_train, y_train)
pred_sv = svr.predict(X_test)
r2_sv = r2_score(y_test, pred_sv)
mse_sv = mean_squared_error(y_test, pred_sv)


#RandomForest 
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
rfr = RandomForestRegressor(random_state=2022)
rfr.fit(X_train, y_train)
pred_rf = rfr.predict(X_test)
r2_rf = r2_score(y_test, pred_rf) #
mse_rf = mean_squared_error(y_test, pred_rf) #

#XGBRegressor
from xgboost import XGBRegressor
xgr = XGBRegressor(random_state=2022)
xgr.fit(X_train, y_train)
pred_xg = xgr.predict(X_test)
r2_xg = r2_score(y_test, pred_xg)
mse_xg = mean_squared_error(y_test, pred_xg)

df = pd.DataFrame({
    'y_test':y_test, 'LR':pred_lr, 'DT':pred_dt, 'SVM':pred_sv, 'RF':pred_rf, 'XG':pred_xg
})
print(df.head())