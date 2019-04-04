import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt


rng = np.random.RandomState(1)

x = np.sort(5 * rng.rand(200, 1))
y = np.sin(x)
y[rng.randint(0, 200, 10)] += rng.rand()

x_test = np.arange(0.0, 5.0, 0.01).reshape(-1, 1)

tree_regr = tree.DecisionTreeRegressor(max_depth=1)
tree_regr.fit(x, y)
pred_y = tree_regr.predict(x_test)


plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="data")
plt.plot(x_test, pred_y, color="yellowgreen", label="max_depth=5", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()