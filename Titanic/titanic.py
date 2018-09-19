from keras.models import *
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd



def load_data():
    train_data = pd.read_csv("data/train.csv", index_col=0)
    test_data = pd.read_csv("data/test.csv", index_col=0)
    return train_data, test_data


def pre_proc(data):
    # here we drop unnecessary attributes.
    data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

    # fill the missing data.
    data[['Age']] = data[['Age']].fillna(value = data[['Age']].mean())
    data[['Fare']] = data[['Fare']].fillna(value = data[['Fare']].mean())
    data[['Embarked']] = data[['Embarked']].fillna(value=data['Embarked'].value_counts().idxmax())

    # convert categorical features into numeric ones
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)

    # convert Embarked to one-hot
    enbarked_one_hot = pd.get_dummies(data['Embarked'], prefix='Embarked')
    data = data.drop('Embarked', axis=1)
    data = data.join(enbarked_one_hot)

    return data


def proc_train_data(train_data):
    X = train_data.drop(['Survived'], axis=1).values.astype(float)
    scale = StandardScaler()
    X = scale.fit_transform(X)

    Y = train_data['Survived'].values
    return X, Y


def fnn(input_dim = (9, )):
    inputs = Input(input_dim);

    fc = Dense(128, activation='relu', kernel_initializer='truncated_normal', bias_initializer='zeros')(inputs)

    for i in range(0, 5):
        drop = Dropout(0.5)(fc)
        fc = Dense(64, activation='relu', kernel_initializer='truncated_normal', bias_initializer='zeros')(drop)

    outputs = Dense(1, activation='relu', kernel_initializer='normal')(fc)

    model = Model(input=inputs, output=outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


train_data, test_data = load_data()
train_data = pre_proc(train_data)
X, Y = proc_train_data(train_data)
model = fnn()
model.fit(X, Y, epochs=500, verbose=2)
#X_test = pre_proc(test_data)
#scale = StandardScaler()
#X_test = scale.fit_transform(X_test)
#Y_test = model.predict(X_test)
#print(Y_test)