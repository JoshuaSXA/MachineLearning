from sklearn import datasets
from sklearn import tree
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris()

train_x, test_x, train_y, test_y = train_test_split(iris.data, iris.target, test_size=0.2)


d_tree = tree.DecisionTreeClassifier()
d_tree.fit(train_x, train_y)

predict_y = d_tree.predict(test_x)

correct_counter = 0
for i in range(len(predict_y)):
    if predict_y[i] == test_y[i]:
        correct_counter = correct_counter + 1
print("Accuracy in test datasets is %f" % (correct_counter / len(predict_y)))


