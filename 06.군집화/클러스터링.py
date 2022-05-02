''' K-MEANS는 군집 중심점이라는 특정한 임의의 지점을 선택해 해당 중심에 가장 가까운 포인트들을 선택하는 군집화 기법. 다음 중심점은 선택된 포인트의 평균 지점으로 이동.
이동된 중심점에서 다시 가까운 포인트를 선택, 다시 중심점을 평균 지점으로 이동하는 프로세스를 반복적으로 수행.

장점 : 일반적인 군집화에 가장 많이 사용한다. 알고리즘이 이해하기도 쉽고 간결하다. 
단점 : 반복 횟수가 많을 경우 수행 시간이 매우 느려진다. 몇 개의 군집을 선택해야할 지 가이드하기가 어렵다. 거리 기반 알고리즘이기 때문에 속성 개수가 많을 경우 군집화 정확도가 떨어진다.(이럴 때에는 차원축소)
'''

from random import random
from re import U
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import rand
from sklearn import cluster

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler #표준화
from sklearn.decomposition import PCA #고차원의 데이터를 주성분으로 저차원 공간으로 변환 

iris = load_iris()
iris_std = StandardScaler().fit_transform(iris.data)
print(iris_std.shape) #150행 4열 구조 (4차원)
pca = PCA(n_components=2) #2차원으로 변환 
iris_pca = pca.fit_transform(iris_std)
print(iris_pca.shape) #150행 2열 구조 (2차원)

from sklearn.cluster import KMeans
print(KMeans().get_params())
''' max_iter : 최대 군집화 반복 횟수, n_clusters : 군집화 그룹 수, '''

kmeans = KMeans(n_clusters=3, random_state=20)
kmeans.fit(iris_std) #비지도학습이므로 y값은 안줘도 된다.

print(kmeans.labels_) #군집화 라벨링
print(np.unique(kmeans.labels_, return_counts=True)) #군집화 결과 고유값

df = pd.DataFrame({
    'PC1':iris_pca[:,0], 'PC2':iris_pca[:,1],'target':iris.target, 'cluster':kmeans.labels_
})

print(df.head(55).tail(5)) #상위 55개 추출 그 중 끝에서 5개 출력

df_group=df.groupby(['target','cluster'])[['PC1']].count()
print(df_group)

#군집 결과 시각화
markers = ['^', 's', 'o']

for i, marker in enumerate(markers):
    #(0,'^'), (1,'s'), (2.'o')
    x_data = df[df.cluster == i]['PC1'] #조건 1 클러스터열 0,1,2 만족하는 것 
    y_data = df[df.cluster == i]['PC2'] #조건 2 클러스터열 0,1,2, 만족하는 것
    plt.scatter(x_data, y_data, marker=marker, label='cluster'+str(i))

plt.legend()
plt.xlabel('PCA Component1'), plt.ylabel('PCA Component2')
plt.show()

#원본 데이터와 군집화된 데이터 비교
plt.figure(figsize=(12,4))

for k, column in enumerate(['target', 'cluster']):
    plt.subplot(1,2,k+1) #subplot 배열 (0,'target) (1,'cluster)
    for i, marker in enumerate(markers):
        x_data = df[df[column]==i]['PC1']
        y_data = df[df[column]==i]['PC2']

        if k == 0 : #원본 데이터면
            plt.title('Original Data', fontsize=15)
            plt.scatter(x_data,y_data, marker=marker, label=iris.target_names[i])
        
        else:
            plt.title('Cluster Data', fontsize=15)
            plt.scatter(x_data, y_data, marker=marker, label='cluster'+str(i))
    
    plt.legend()
    plt.xlabel('PCA component1'), plt.ylabel('PCA component2')
plt.show()

'''비교 그래프를 보면 알 수 있듯이 실제 데이터와 군집화된 데이터에는 차이가 있다. Kmeans의 단점이 이 부분에 있다.'''




'''kmeans vs mean shift (평균이동)
kmeans는 데이터의 평균 거리 중심으로 이동한다. 반면에 평균이동은 중심을 데이터가 모여 있는 밀도가 가장 높은 곳으로 이동시킨다
대역폭 크기로 움직이기 때문에 별도로 군집화 수를 지정할 필요가 없다. '''

from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

X, y = make_blobs(n_samples=200, n_features=2, centers=3, cluster_std=0.8, random_state=0)

meanshift = MeanShift(bandwidth=0.9)
cluster_labels = meanshift.fit_predict(X)
print('cluster labels unique :', np.unique(cluster_labels) )

meanshift = MeanShift(bandwidth=1)
cluster_labels = meanshift.fit_predict(X)
print('cluster labels unique :', np.unique(cluster_labels) )

''' bandwidth best estimator '''
from sklearn.cluster import estimate_bandwidth
bandwidth = estimate_bandwidth(X, quantile=0.2)
print('best_bandwidth: ' , round(bandwidth, 3))

clusterdf = pd.DataFrame(data=X, columns=['ftr1', 'ftr2'])
clusterdf['target'] = y
meanshift = MeanShift(bandwidth=1.444)
cluster_labels = meanshift.fit_predict(X)

clusterdf['meanshift_labels'] = cluster_labels
centers = meanshift.cluster_centers_
unique_labels = np.unique(cluster_labels)
markers = ['o','s','^','x','*']

for label in unique_labels:
    label_clsuter= clusterdf[clusterdf['meanshift_labels']==label]
    center_x_y = centers[label]
    plt.scatter(x=clusterdf['ftr1'], y = clusterdf['ftr2'], edgecolors='k', marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=200, color='white', edgecolors='white', alpha=0.9, marker=markers[label])
    plt.scatter(x=center_x_y[0], y=center_x_y[1], s=70, color='white', edgecolors='k', alpha=0.9, marker='$%d$' % label)
plt.show()