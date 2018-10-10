import tensorflow as tf
import numpy as np
import pandas as pd
import pickle

train_data = {b'data':[], b'labels':[]}

#加载训练数据
for i in range(5):
    with open("cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
        data = pickle.load(file, encoding='bytes')
        train_data[b'data'] += list(data[b'data'])
        train_data[b'labels'] += data[b'labels']

# 加载测试数据
with open("cifar-10-batches-py/test_batch", mode='rb') as file:
    test_data = pickle.load(file, encoding='bytes')

# 定义全局变量
NUM_LABELS = 10
FC_SIZE = 512
BATCH_SIZE = 100
lamda = 0.004 #正则化系数，未使用

sess = tf.InteractiveSession() #区分与tf.Session()

# 用来初始化权重
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 卷积层偏置初始化为常数0.1
def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

# 定义卷积操作，卷积步长为1，padding = ‘SAME’，表示全0填充
def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

# 定义最大池化操作，尺寸为2，步长为2，全0填充
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding='SAME')

# 对输入进行占位操作，输入为BATCH*3072向量，输出为BATCH*10的向量
x = tf.placeholder(tf.float32, [None, 3072])
y_ = tf.placeholder(tf.float32, [None, NUM_LABELS])
# 对输入进行reshape，转换成3*32*32格式
x_image = tf.reshape(x, [-1, 3, 32, 32])
#转置操作
x_image = tf.transpose(x_image, [0, 2, 3, 1])

# 第一层卷积，滤波器参数5*5*3，32个
w_conv1 = weight_variable([5, 5, 3, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# 第二层卷积，滤波器参数为5*5*32，64个
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# 将8*8*64三维向量转换成为一维向量
h_pool2_flat = tf.reshape(h_pool2, [-1, 8 * 8 * 64])

# 第一层全连接层
w_fc1 = weight_variable([8 * 8 * 64, FC_SIZE])
b_fc1 = bias_variable([FC_SIZE])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
# 对隐藏层使用dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 第二层全连接
w_fc2 = weight_variable([FC_SIZE, NUM_LABELS])
b_fc2 = bias_variable([NUM_LABELS])
y = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)
#L2正则化
w1_loss = lamda * tf.nn.l2_loss(w_fc1)
w2_loss = lamda * tf.nn.l2_loss(w_fc2)
# 交叉熵损失
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices = [1]))

# 总损失
loss = w1_loss + w2_loss + cross_entropy

# 用AdamOptimizer优化器训练
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 计算准确率
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 开始训练
sess.run(tf.initialize_all_variables())




# 对数据范围为0-255的训练数据做归一化处理使其范围为0-1，并将list转成numpy向量
x_train = np.array(train_data[b'data']) / 255
# 将训练输出标签变成one_hot形式并将list转成numpy向量
y_train = np.array(pd.get_dummies(train_data[b'labels']))

# 对数据范围为0-255的测试数据做归一化处理使其范围为0-1，并将list转成numpy向量
x_test = test_data[b'data'] / 255
# 将测试输出标签变成one_hot形式并将list转成numpy向量
y_test = np.array(pd.get_dummies(test_data[b'labels']))

# 训练
for i in range(10000):
    # 100条数据为1个batch，轮流训练
    start = i * BATCH_SIZE % 50000
    train_step.run(feed_dict = {x: x_train[start: start + BATCH_SIZE],
                                    y_: y_train[start: start + BATCH_SIZE], keep_prob: 0.5})
    # 每迭代100次在前200条个测试集上测试训练效果
    if i % 100 == 0:
        # 测试准确率
        train_accuracy = accuracy.eval(feed_dict={x: x_test[0: 200], y_: y_test[0: 200], keep_prob: 1.0})
        # 该次训练的损失
        loss_value = cross_entropy.eval(feed_dict = {x: x_train[start: start + BATCH_SIZE], y_: y_train[start: start + BATCH_SIZE], keep_prob: 0.5})
        print("step %d, trainning accuracy， %g loss %g" % (i, train_accuracy, loss_value))

# 测试
test_accuracy = accuracy.eval(feed_dict = {x: x_test, y_: y_test, keep_prob: 1.0})
print("test accuracy %g" % test_accuracy)







    