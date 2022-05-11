'''얼마나 효율적으로 됐는지 평가할 타깃 레이블이 없는 경우가 대다수. 
또한 군집화는 엄연히 분류와는 다르다. 분류를 보다 더 세분화하여 군집화할 수 있기 때문이다. 
이런 군집화를 평가하기 위해서는 실루엣 분석이 대표적으로 사용된다.

실루엣 분석은 각 군집 간의 거리가 얼마나 효율적으로 분리돼 있는지를 나타낸다. 
동일 군집 간 거리는 서로 가깝고 잘 뭉쳐있고 다른 군집과는 떨어져있다.

실루엣 분석은 실루엣 계수(silhouette coefficient)라는 개별 데이터가 가지는 군집화 지표를 기반으로 분석.
같은 군집 내 다른 데이터와의 거리의 평균을 a(i)
다른 군집 다른 데이터와의 거리의 평균을 b(i)
계수 s(i) = (b(i)-a(i))/max(a(i),b(i)) 
-1 <= s(i) <= 1  
1에 가까울수록 다른 군집과 더 멀리 떨어져있다.
0에 가까울수록 다른 군집과 더 가깝다는 것 
음수는 아예 다른 군집에 데이터 포인트가 할당되었음을 의미
sklearn.metrics.silhouette_samples(data, label) -> 전체 데이터 각각에 대한 계수값 반환
sklearn.metrics.silhouette_score(data, label) -> samples mean return, higher score is better(but not always)
좋은 군집화의 조건 1. 평균스코어가 0~1 사이 값. 1에 가까울수록 좋다 
좋은 군집화의 조건 2. 개별 군집의 평균값의 편차가 크지 않아야 한다. (균일성)'''

from base64 import standard_b64decode
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
iris = load_iris()
iris_std = StandardScaler().fit_transform(iris.data)
kmeans = KMeans(n_clusters=3, random_state=20)
kmeans.fit(iris_std)

df = pd.DataFrame(iris.data, columns=['sl','sw','pl','pw'])
df['target'] = iris.target
df['cluster'] = kmeans.labels_

#Silhouette coeffiecient
sil_samples = silhouette_samples(iris_std, kmeans.labels_)
df['coefficient'] = sil_samples

#silhouette_score(iris_std, kmeans.labels_)
#df['coefficient'].mean()

print(df.groupby('cluster')['coefficient'].mean())

#실루엣 계수 시각화로 평균의 함점 피하기

from visualize import visualize_silhouette
visualize_silhouette([2,3,4,5], iris.data) # ([2.3.4.5])는 군집이 2,3,4,5개 일때 각각 계산, X_features 대입 
plt.show()

''' 실루엣 계수 평가 방법은 직관적 이해는 쉽지만 군집 내, 군집 간 거리 계산을 중복해서 해야하므로 시간이 오래걸리게 된다. 
특히 몇 만 건 이상의 데이터에 대해 계수 평가를 개인용 컴퓨터로 할 경우 메모리 부족 에러가 발생할 수 있다.
이 경우 샘플링을 하여 계수를 평가한 것이 대안이 될 수 있다.'''
