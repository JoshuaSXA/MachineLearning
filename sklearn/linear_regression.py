import numpy as np
from numpy import random
from sklearn import linear_model

reg = linear_model.LinearRegression()

# 生成训练数据
x = np.arange(start=0, stop=100, step=1, dtype=np.float32)
y = x * 2 + 1.0
train_x = (x + random.randn(100)).reshape(-1, 1)
train_y = y + random.randn(100)

# 拟合数据
reg.fit(train_x, train_y)

print(reg.coef_)
