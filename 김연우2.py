import imp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# 과제 2 : 유방암 데이터 - 차원축소, 군집화
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA

#데이터 전처리
from sklearn.preprocessing import StandardScaler
cancer = load_breast_cancer() # 인스턴스화 
cancer_std = StandardScaler().fit_transform(cancer.data)

#데이터 split, 분류
def pca_acc(X,y):
    X_train,X_test,y_train,y_test = train_test_split(
        X,y, stratify=y, random_state=2022
    )
    rfc = RandomForestClassifier(random_state=2022)
    rfc.fit(X_train, y_train)
    score = rfc.score(X_test, y_test)
    return score

#군집화
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=2022)
kmeans.fit(cancer_std)

#차원축소, 군집화 결과 출력
for i in [10,5,2]:
    pca = PCA(i)
    cancer_pca = pca.fit_transform(cancer_std)
    explained = pca.explained_variance_ratio_.sum()
    acc = pca_acc(cancer_pca, cancer.target)
    print(f'PCA({str(i)})의 설명력 : {explained:.4f}\nPCA({str(i)})의 정확도 : {acc:.4f}')
    df = pd.DataFrame({
    
    'PC1':cancer_pca[:,0], 'PC2':cancer_pca[:,1],
    'target':cancer.target, 'cluster':kmeans.labels_})
    print(f'PCA{str(i)}의 군집화 결과 :', df.tail(5))


# 2차원 PCA 시각화
print(df.head(10))

markers = ['^','s']
plt.figure(figsize=(12,4))
for k, column in enumerate(['target','cluster']):
    plt.subplot(1, 2, k+1)
    for i, marker in enumerate(markers):
        x_data = df[df[column] == i]['PC1']
        y_data = df[df[column] == i]['PC2']
        if k == 0:
            plt.title('Original data', fontsize=15)
            plt.scatter(x_data, y_data, marker=marker, label=cancer.target_names[i])
        else:
            plt.title('Cluster data', fontsize=15)
            plt.scatter(x_data, y_data, marker=marker, label='cluster'+str(i))

    plt.legend()
    plt.xlabel('PCA Component 1'), plt.ylabel('PCA Component 2')
plt.show()
