#결정트리(DecisionTree)

from multiprocessing.pool import IMapUnorderedIterator
from xml.sax.handler import feature_external_ges
from sklearn.datasets import load_iris #내장된 자료 셋
from sklearn.model_selection import train_test_split  #데이터 분리 클래스
from sklearn.tree import DecisionTreeClassifier #분류기

iris = load_iris()
X_train, X_test, y_train,y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=11
)

dtc = DecisionTreeClassifier(random_state=156)
dtc.fit(X_train, y_train)

#모델 시각화
from sklearn.tree import export_graphviz
export_graphviz(
    dtc, out_file='tree.dot',
    feature_names = iris.feature_names, class_names=iris.target_names,
    impurity=True, filled=True
)


import graphviz
with open('tree.dot') as file:
    dot_graph = file.read()

graphviz.Source(dot_graph)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(14,12))
plot_tree(dtc, feature_names=iris.feature_names, class_names=iris.target_names, filled = True)
plt.show()

# 제약조건 
dtc2 = DecisionTreeClassifier(max_depth=3, random_state=156)
dtc2.fit(X_train,y_train)

export_graphviz(
    dtc2, out_file='tree2.dot',
    feature_names=iris.feature_names, class_names=iris.target_names,impurity=True, filled=True
)

with open('tree2.dot') as file:
    dot_graph = file.read()

graphviz.Source(dot_graph)
