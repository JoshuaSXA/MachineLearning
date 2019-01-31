import os
import torch
import torch.utils.data as Data
import pickle
import numpy as np

class CifarDataLoader(object):
    def __init__(self, data_path, train_file_prefix, test_filename):
        torch.set_default_tensor_type(torch.DoubleTensor)
        self._data_path = data_path
        self._train_file_prefix = train_file_prefix
        self._test_filename = test_filename
        self._train_dataset = None
        self._test_dataset = None

    def load_file(self, filename):
        with open(os.path.join(self._data_path, filename), 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        x = np.array(data['data'].astype(float))
        y = np.array(data['labels'])
        return x, y

    def load_data(self):
        x = []
        y = []
        for i in range(1, 6):
            filename = self._train_file_prefix + str(i)
            x_data, y_label = self.load_file(filename)
            x.append(x_data)
            y.append(y_label)
        x_train = np.concatenate(x)
        y_train = np.concatenate(y)
        x_test, y_test = self.load_file(self._test_filename)
        # Data mean subtraction
        x_train -= np.mean(x_train, axis=0)
        x_test -= np.mean(x_test, axis=0)
        # Data reshape
        x_train = torch.from_numpy(np.reshape(x_train, (-1, 3, 32, 32)))
        x_test = torch.from_numpy(np.reshape(x_test, (-1, 3, 32, 32)))
        y_train = torch.from_numpy(y_train).long()
        y_test = torch.from_numpy(y_test).long()
        # Generate torch dataset
        self._train_dataset = Data.TensorDataset(x_train, y_train)
        self._test_dataset = Data.TensorDataset(x_test, y_test)

    def get_data_loader(self, test=False, batch_size=100, shuffle=False):
        # Initiate data_loader
        data_loader = None
        if not test and self._train_dataset:
            data_loader = Data.DataLoader(self._train_dataset, batch_size=batch_size, shuffle=shuffle)
        elif test and self._test_dataset:
            data_loader = Data.DataLoader(self._test_dataset)
        return data_loader


# if __name__ == '__main__':
#     data_loader = CifarDataLoader('./data/cifar-10-batches-py/', 'data_batch_', 'test_batch')
#     data_loader.load_data()
#     train_loader = data_loader.get_data_loader(batch_size=100, shuffle=True)
#     print(train_loader)