#로지스틱 회귀 분류

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#sigmoid 함수

def sigmoid(x, a=1, b=0):
    return (1. / (1+np.exp(-a*(x-b)))) 

xs = np.linspace(-5,5,1001)
ys = sigmoid(xs)

plt.plot (xs,ys,label='sigmoid')
plt.plot(xs, ys*(1-ys), label='derivative')
plt.title('Sigmoid function')
plt.yticks([0,0.5,1])
plt.grid()
plt.legend()
plt.show()

# a값 기울기 변경해보기  (b는 축 이동이니까 안함)
y3 = sigmoid(xs, a=3)
y_half = sigmoid(xs, a=0.5)
plt.plot (xs,ys,label='sigmoid')
plt.plot (xs,y3,label='sigmoid *3')
plt.plot (xs,y_half,label='sigmoid *0.5')

plt.title('Sigmoid function')
plt.yticks([0,0.5,1])
plt.grid()
plt.legend()
plt.show()


#이진분류
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.data.shape

from sklearn.preprocessing import StandardScaler
cancer_std = StandardScaler().fit_transform(cancer.data)
X_train, X_test, y_train, y_test = train_test_split(
    cancer_std, cancer.target, strartify = cancer.target, random_state=2022
)

#Logistic 회귀
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression(random_state=2022)
lrc.fit(X_train,y_train) #학습

print(lrc.coef_) #계수
print(lrc.intercept_ )
print(X_test[0])


# X_test[0]가 1이 될 확률 

val = np.dot(lrc.coef_, X_test[0]) + lrc.intercept_ #(1,30) * (30, )행렬 연산
print(val)
print(sigmoid(val))
print(lrc.predict_proba(X_test[:5]))


#다중 분류
from sklearn.datasets import load_wine
wine = load_wine()
wine_std = StandardScaler().fit_transform(wine.data)
X_train,X_test,y_train,y_test = train_test_split(
    wine_std, wine.target, stratify=wine.target, random_state=2022
)

