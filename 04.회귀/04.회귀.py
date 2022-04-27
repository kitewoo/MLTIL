import matplotlib.pyplot as plt
import pandas as pd 


df = pd.read_csv('http://www.randomservices.org/random/data/Galton.txt', sep='\t')

#filtering : 아들만, 컬럼명 키 -> 아들, 
df = df[df.Gender == 'M']
df = df[['Father', 'Height']]
df.rename(columns={'Height':'Son'}, inplace=True)
df.reset_index(drop=True, inplace=True)
df = df*2.54

print(df.head())

# Father가 독립변수, Son이 종속변수
plt.scatter(df.Father, df.Son)
plt.grid()
plt.xlabel("Fathers' heihgt (cm)")
plt.ylabel("Sons' height (cm)")
plt.title('Father VS Son', fontsize=15)
plt.show()

#회귀선 구하고 그리기
#np.linalg.lstsq()
#np.polyfit() 

import  numpy as np 

height, bias = np.polyfit(df.Father, df.Son, 1) #X, Y, 차수
print(height, bias)

#회귀식을 2차식으로 
print(np.polyfit(df.Father, df.Son, 2))

xs = np.array([156,201])
ys = xs*height + bias

plt.scatter(df.Father, df.Son)
plt.grid()
plt.plot(xs,ys, color='r')
plt.xlabel("Fathers' heihgt (cm)")
plt.ylabel("Sons' height (cm)")
plt.title('Father VS Son', fontsize=15)
plt.show()



import seaborn as sns
sns.regplot(x='Father', y='Son', data=df)
plt.show()

#Scikit-learn으로 회귀식 구하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.get_params()
#회귀도 지도학습의 한 형태이다. 학습 메소드는 결국 lr.fit이다.
print(df.shape) #(465,2)

lr.fit(df[['Father']], df.Son)
lr.fit(df.Father.values.reshape(-1,1), df.Son.values)

# coef 계수
lr.coef_

# intercept(절편), bias
lr.intercept_

# 잔차 제곱의 합
lr._residues

# 평가 - R.squared (설명력, 결정계수)
lr.score(df.Father.values.reshape(-1,1), df.Son.values)

from sklearn.metrics import r2_score
pred = lr.predict(df.Father.values.reshape(-1,1))
r2_score(df.Son.values, pred)

