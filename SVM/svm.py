import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import time

data = pd.read_csv("dataset", header=None)
data = data.values
x = data[1:,1:]
print(x.shape)
y = data[1:,0]
print(y.shape)
acc = 0
start = time.time()

for i in range(100):
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.5)
    clf = svm.SVC(kernel="linear", C=2)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    acc = acc + metrics.accuracy_score(y_test, y_pred)
print(acc/100.0)
end = time.time()
print("runnning time:",end-start)
