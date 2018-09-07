from keras.datasets import cifar10
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.utils import *
import numpy as np
import pickle



(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

Y_train = np_utils.to_categorical(Y_train, 10)
Y_test = np_utils.to_categorical(Y_test, 10)

# 读取数据函数
def load_data():
    # 初始化训练数据
    train_data = {b'data':[], b'labels':[]}
    # 加载训练数据
    for i in range(5):
        with open("cifar-10-batches-py/data_batch_" + str(i + 1), mode='rb') as file:
            data = pickle.load(file, encoding='bytes')
            train_data[b'data'] += list(data[b'data'])
            train_data[b'labels'] += data[b'labels']
    train_data[b'data'] = np.array(train_data[b'data']) / 255
    # 加载测试数据
    with open("cifar-10-batches-py/test_batch", mode='rb') as file:
        test_data = pickle.load(file, encoding='bytes')
    data = {'train_data':train_data, 'test_data':test_data}
    return data

# 定义GoogLeNet网络结构
def google_net(input_size = (32,32,3)):
    # 输入层
    inputs = Input(input_size)

    conv1 = Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    bn1 = BatchNormalization(axis=1, epsilon=1e-06, mode=0)(conv1)
    pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(bn1)

    # conv2 = Conv2D(16, 1, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    bn2 = BatchNormalization(axis=1, epsilon=1e-06, mode=0)(conv2)
    pool2 = MaxPool2D(pool_size=(2, 2), padding='valid')(bn2)

    conv3 = Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    bn3 = BatchNormalization(axis=1, epsilon=1e-06, mode=0)(conv3)
    pool3 = MaxPool2D(pool_size=(2, 2), padding='valid')(bn3)

    conv4 = Conv2D(32, 3, strides=1, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    bn4 = BatchNormalization(axis=1, epsilon=1e-06, mode=0)(conv4)
    pool4 = MaxPool2D(pool_size=(2, 2), padding='valid')(bn4)

    # 全连接层
    drop1 = Dropout(0.5)(pool4)
    flat_layer = Flatten()(drop1)
    fc1 = Dense(100, activation='relu')(flat_layer)
    drop2 = Dropout(0.5)(fc1)
    fc2 = Dense(10, activation='softmax')(drop2)

    model = Model(input=inputs, output=fc2)
    model.compile(optimizer=SGD(lr=0.01, decay=1e-06, momentum=0.9, nesterov=True), loss='categorical_crossentropy',  metrics=['accuracy'])
    return model



model = google_net()
model.fit(X_train, Y_train, batch_size=100, nb_epoch=50, validation_data=(X_test, Y_test))














