from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# load data from datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

# divide the datasets into train sets and test sets
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.2)

# define the KNN model
knn = KNeighborsClassifier()
# train
knn.fit(train_x, train_y)
# predict
predict_y = knn.predict(test_x)

correct_counter = 0
for i in range(len(predict_y)):
    if predict_y[i] == test_y[i]:
        correct_counter = correct_counter + 1
print("Accuracy in test datasets is %f" % (correct_counter / len(predict_y)))

