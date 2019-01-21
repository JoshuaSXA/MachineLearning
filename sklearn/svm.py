from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn import svm

# load data from datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

# divide the datasets into train sets and test sets
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.2)

# define SVM
svm_svc_clf = svm.SVC()
svm_nusvc_clf = svm.NuSVC()
svm_linear_clf = svm.LinearSVC()

# train
svm_svc_clf.fit(train_x, train_y)
svm_nusvc_clf.fit(train_x, train_y)
svm_linear_clf.fit(train_x, train_y)

# predict
svc_y = svm_svc_clf.predict(test_x)
nusvc_y = svm_nusvc_clf.predict(test_x)
linear_y = svm_linear_clf.predict(test_x)

# accuracy
svc_counter = 0
nusvc_counter = 0
linear_counter = 0
for i in range(len(test_y)):
    if test_y[i] == svc_y[i]:
        svc_counter = svc_counter + 1
    if test_y[i] == nusvc_y[i]:
        nusvc_counter = nusvc_counter + 1
    if test_y[i] == linear_y[i]:
        linear_counter = linear_counter + 1

print("SVM accuracy is %f, NuSVM accuracy is %f, Linear SVM accuracy is %f" % (svc_counter / len(test_y), nusvc_counter / len(test_y), linear_counter / len(test_y)))


