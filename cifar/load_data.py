import numpy as np
from keras.utils import *
import os



def load_data_batch(filename, data_type = 'float64'):
    path = os.path.join('cifar-10-batches-py', filename)
    data = np.load(path, encoding="latin1")
    data['data'] = data['data'].astype(data_type)

    data['data'] = data['data'] / 255.0
    data['labels'] = np_utils.to_categorical(np.array(data['labels']), 10)

    return data



def data_reshape(data):
    data = data.reshape(data.shape[0], 3, 32,32).transpose(0,2,3,1)
    print(data.shape)

    return data



# 读取数据函数
def load_cifar10_data():

    train_data = {'data':[], 'labels':[]}

    for i in range(5):
        data = load_data_batch('data_batch_%d' % (i + 1))
        train_data['data'].append(data['data'])
        train_data['labels'].append(data['labels'])


    test_data = load_data_batch('test_batch')

    train_data['data'] = np.concatenate(train_data['data'], axis=0)
    train_data['labels'] = np.concatenate(train_data['labels'], axis=0)

    train_data['data'] = data_reshape(train_data['data'])
    test_data['data'] = data_reshape(test_data['data'])

    return train_data, test_data