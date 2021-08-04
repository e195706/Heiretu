import pandas as pd
from pandas import DataFrame
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import ensemble
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

start = time.time()
mnist = fetch_openml('mnist_784', cache=False)

x = mnist.data
y = mnist.target
print(x.shape)
print(y.shape)

x = x/255

x = pd.DataFrame(x)
y = pd.DataFrame(y)
#x = x.drop(range(65000))
#y = y.drop(range(65000))
print(x.shape)
print(y.shape)

X_train_valid, X_test, y_train_valid, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_valid, y_train_valid, test_size=0.2, random_state=0)

#SVM(SVC)
classfication = svm.SVC()
classfication.fit(X_train,y_train)
prediction = classfication.predict(X_test)
collect_score = metrics.accuracy_score(y_test, prediction)
train_s = classfication.predict(X_valid)
print("SVCの正解率 =", collect_score)
#KNN
knn_classfication = KNeighborsClassifier()
knn_classfication.fit(X_train,y_train)
knn_prediction = knn_classfication.predict(X_test)
knn_collect_score = metrics.accuracy_score(y_test, knn_prediction)
train_k = knn_classfication.predict(X_valid)
print("KNNの正解率 =", knn_collect_score)
#ロジスティック回帰
Log_classfication = LogisticRegression()
Log_classfication.fit(X_train,y_train)
Log_prediction = Log_classfication.predict(X_test)
Logistic_collect_score = metrics.accuracy_score(y_test, Log_prediction)
train_l = Log_classfication.predict(X_valid)
print("ロジスティック回帰の正解率 =", Logistic_collect_score)

y_pred_s = prediction
y_pred_k = knn_prediction
y_pred_l = Log_prediction

stack_train = np.column_stack((train_s, train_k, train_l))
stack_clf = ensemble.RandomForestClassifier(n_estimators=100, random_state=0)
stack_clf.fit(stack_train, y_valid)
stack_last_train = np.column_stack((y_pred_s, y_pred_k, y_pred_l))
stack_pred = stack_clf.predict(stack_last_train)
print("ランダムフォレストの正解率", metrics.accuracy_score(y_test,stack_pred))

elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(round(elapsed_time)) + "[sec]")