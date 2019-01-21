from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB

# load data from datasets
iris = datasets.load_iris()
iris_x = iris.data
iris_y = iris.target

# divide the datasets into train sets and test sets
train_x, test_x, train_y, test_y = train_test_split(iris_x, iris_y, test_size=0.2)

# define naive bayes model
gnb = GaussianNB()
gnb.fit(train_x, train_y)
predict_y = gnb.predict(test_x)

# accuracy
correct_counter = 0
for i in range(len(test_y)):
    if test_y[i] == predict_y[i]:
        correct_counter = correct_counter + 1

print("Gaussian Naive Bayes accuracy is %f" % (correct_counter / len(test_y)))
