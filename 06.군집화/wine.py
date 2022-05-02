
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler 
from sklearn.decomposition import PCA

wine = load_wine() #dict data : 2 dimension (wine, features)
#StandardScaler
wine_std = StandardScaler().fit_transform(wine.data) #nd_array, 
print(wine_std)

pca = PCA(n_components=2)
wine_pca = pca.fit_transform(wine_std) #stand -> nd_array pca 
print(wine_pca)
# Clustering
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=20)
kmeans.fit(wine_std) #3 cluster

# DataFrame(pca0 ,pca1, target(original), cluster(fit))
df = pd.DataFrame({'PC1': wine_pca[:,0], 'PC2':wine_pca[:,1], 'target':wine.target, 'cluster':kmeans.labels_})

kmeans.cluster_centers_  # 군집 중심 좌표 

markers = ['^','s','o']
plt.figure(figsize=(12,4))
for k, column in enumerate(['target','cluster']):
    plt.subplot(1, 2, k+1)
    for i, marker in enumerate(markers):
        x_data = df[df[column] == i]['PC1']
        y_data = df[df[column] == i]['PC2']
        if k == 0:
            plt.title('Original data', fontsize=15)
            plt.scatter(x_data, y_data, marker=marker, label=wine.target_names[i])
        else:
            plt.title('Cluster data', fontsize=15)
            plt.scatter(x_data, y_data, marker=marker, label='cluster'+str(i))
            plt.scatter(x_data.mean(), y_data.mean(), marker='*', c='black', s=100)

    plt.legend()
    plt.xlabel('PCA Component 1'), plt.ylabel('PCA Component 2')
plt.show()

