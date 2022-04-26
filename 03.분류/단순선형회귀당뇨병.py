import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#data preprocessing
from sklearn.datasets import load_diabetes
diabetes  = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target


'''import seaborn as sns
_, ax = plt.subplots(2, 5, figsize=(18,8))

for i in df.columns[:-1]:
    row, col = i//5, i%5
    sns.regplot(x=i, y='target', data=df, ax=axs[row][col])
    plt.show()'''

from sklearn.linear_model import LinearRegression
lr = LinearRegression()

for feature in df.columns[:-1]:
    X = df[feature].values.reshape(-1,1)
    lr.fit(X, diabetes.target)
    score = lr.score(X, diabetes.target)
    print(f'{feature}:{score:.4f}')


### BMI vs target : 훈련/테스트 데이터셋 분리 size = 0.1
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    df.bmi.values.reshape(-1,1), diabetes.target, test_size=0.1, random_state=2022
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


lr = LinearRegression()
lr.fit(X_train, y_train)

# 계수, 절편, 잔차 제곱의 합
lr.coef_
lr.intercept_
lr._residues

from sklearn.metrics import r2_score
pred = lr.predict(X_test)
print(r2_score(y_test, pred))


#Mean Squared Error
from sklearn.metrics import mean_squared_error #feature간 비교할 때 사용
mse = mean_squared_error(y_test, pred)
rmse = np.sqrt(mse)
print(mse, rmse)


#시각화 

plt.scatter(X_train, y_train, label='train')
plt.scatter(X_test, y_test, marker='^', label='test')

plt.grid()
plt.xlabel("BMI")
plt.ylabel("Diabetes")
plt.title('BMI VS Diabetes', fontsize=15)
plt.show()



# BP vs target
X_train2, X_test2, y_train2, y_test2 = train_test_split(
    df.bp.values.reshape(-1,1), diabetes.target, test_size=0.1, random_state=2022
)

lr2 = LinearRegression()
lr2.fit(X_train2, y_train2)

pred2 = lr2.predict(X_test2)
print(r2_score(y_test,pred), r2_score(y_test2, pred2))
