from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import *
from load_data import load_cifar10_data

# 定义AlexNet网络结构
def my_alexnet(input_size = (32,32,3)):
    # 输入层
    inputs = Input(input_size)

    # 第一层卷积层
    conv1 = Conv2D(32, 3, strides=1, padding='same', kernel_initializer='he_normal')(inputs)
    bn1 = BatchNormalization(axis=1, epsilon=1e-06, mode=0, momentum=0.9)(conv1)
    activation1 = Activation('relu')(bn1)
    pool1 = MaxPool2D(pool_size=(2, 2), padding='valid')(activation1)

    # 第二层卷积层
    conv2 = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool1)
    bn2 = BatchNormalization(axis=1, epsilon=1e-06, mode=0, momentum=0.9)(conv2)
    activation2 = Activation('relu')(bn2)
    pool2 = MaxPool2D(pool_size=(2, 2), padding='valid')(activation2)

    # 第三层卷积层
    conv3 = Conv2D(128, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool2)
    bn3 = BatchNormalization(axis=1, epsilon=1e-06, mode=0, momentum=0.9)(conv3)
    activation3 = Activation('relu')(bn3)
    pool3 = MaxPool2D(pool_size=(2, 2), padding='valid')(activation3)

    # 第四层卷积层
    conv4 = Conv2D(128, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool3)
    bn4 = BatchNormalization(axis=1, epsilon=1e-06, mode=0, momentum=0.9)(conv4)
    activation4 = Activation('relu')(bn4)
    pool4 = MaxPool2D(pool_size=(2, 2), padding='valid')(activation4)

    # 第五层卷积层
    conv5 = Conv2D(64, 3, strides=1, padding='same', kernel_initializer='he_normal')(pool4)
    bn5 = BatchNormalization(axis=1, epsilon=1e-06, mode=0, momentum=0.9)(conv5)
    activation5 = Activation('relu')(bn5)
    pool5 = MaxPool2D(pool_size=(2, 2), padding='valid')(activation5)

    # 全连接层
    drop1 = Dropout(0.5)(pool5)
    flat_layer = Flatten()(drop1)
    fc1 = Dense(100, activation='relu')(flat_layer)
    drop2 = Dropout(0.5)(fc1)
    fc2 = Dense(10, activation='softmax')(drop2)

    # 建立模型
    model = Model(input=inputs, output=fc2)
    # 采用SGD优化器
    model.compile(optimizer=SGD(lr=0.01, decay=1e-06, momentum=0.9, nesterov=True), loss='categorical_crossentropy',  metrics=['accuracy'])
    return model


# 读取cifar-10数据集
train_data, test_data= load_cifar10_data()

# 加载模型
model = my_alexnet()

# 在每个epoch保存训练的模型参数（如果在测试集上表现更好的话）
model_path = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# 训练
model.fit(train_data['data'], train_data['labels'], batch_size=100, nb_epoch=100, callbacks=[checkpoint], validation_data=(test_data['data'], test_data['labels']))














