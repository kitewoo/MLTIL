import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, train_test_split # dataset 분리 클래스
from sklearn.tree import DecisionTreeClassifier # 결정트리 클래스
from sklearn.tree import export_graphviz #결정트리 그림 
from sklearn.tree import plot_tree # tree 그림
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# 데이터 불러오기
df = pd.read_csv('C:/Users/tt/Desktop/machinelearning/03.분류/pima-indians-diabetes.csv', skiprows=9, header=None)

# 데이터 전처리. feature_names로 사용할 컬럼명 지정
df.columns = ['P','G','BP','S','I','BMI','D','Age','target']

# 데이터 전처리. 종속 변수와 독립 변수 분리 as numpy
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(
    X,y,startify=y, test_size=0.2, random_state=2022
)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#고유값을 확인하여 비율 균등한지 확인 (df의 경우 value_counts() 매서드로 가능)
print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))
#DecisionTreeClassifier, SVC
dtc = DecisionTreeClassifier(random_state=2022) #인스턴스화
svc = SVC()

#fit
dtc.fit(X_train,y_train)
svc.fit(X_train,y_train)

#predict X_test -> y_test와 얼마나 맞을려나?
pred_dt = dtc.predict(X_test)
pred_svc = svc.predict(X_test)

#혼란행렬, 정확도, 재현률, 정밀도, f1
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
#decision tree score
print('Confusion_matrix_score: ' , confusion_matrix(y_test, pred_dt))
print('Accurcy_score: ', accuracy_score(y_test, pred_dt))
print('Precision_score: ', precision_score(y_test, pred_dt))
print('recall_score: ', recall_score(y_test, pred_dt))
print('f1_score: ', f1_score(y_test, pred_dt))

#svc score
print('Confusion_matrix_score: ' , confusion_matrix(y_test, pred_svc))
print('Accurcy_score: ', accuracy_score(y_test, pred_svc))
print('Precision_score: ', precision_score(y_test, pred_svc))
print('recall_score: ', recall_score(y_test, pred_svc))
print('f1_score: ', f1_score(y_test, pred_svc))

#DecisionTree : max, min_leaf, min_split
dt_tree = DecisionTreeClassifier(max_depth=3, random_state=156)
dt_tree.fit(X_train, y_train)
plt.figure(figsize=(14,18))
plot_tree(dt_tree, feature_names=df.columns, class_names = 'target', filled=True)
plt.show()

dt_tree = DecisionTreeClassifier(min_samples_split=15, random_state=156)
dt_tree.fit(X_train,y_train)
plt.figure(figsize=(14,18))
plot_tree(dt_tree, feature_names=df.columns, class_names = 'target', filled=True)
plt.show()


dt_tree = DecisionTreeClassifier(min_sample_leaf=15, random_state=156)
dt_tree.fit(X_train,y_train)
plt.figure(figsize=(14,18))
plot_tree(dt_tree, feature_names=df.columns, class_names = 'target', filled=True)
plt.show()

#두 모델의 parameter설정, 베스트 파라미터 찾기
#GridSearchCV 활용
# 두 모델의 parameter 설정

params_dt = {'max_depth': [2, 4, 6],
        'min_samples_leaf': [2, 3, 4, 5, 6],
        'min_samples_split': [2, 4, 6]}

params_svc = {'C':[0.01, 0.1, 1, 10, 100]}
#GridSearchCV 활용

grid_dt = GridSearchCV(dt_tree, params_dt, scoring='accuracy', cv=5)
grid_svc = GridSearchCV(svc, params_svc, scoring='accuracy', cv=5)

grid_dt.fit(X_train, y_train)
grid_svc.fit(X_train,y_train)

#best 파라미터
print("DecisionTree: ", grid_dt.best_params_)
print("SVC: ", grid_svc.best_params_)

#정확도
print(grid_dt.best_estimator_.score(X_test,y_test))
print(grid_svc.best_estimator_.score(X_test,y_test))