import numpy as np

import clean as c
import os
# from train import Model
from model import Model, TF_Model, LR_Model

from sklearn.utils import shuffle

def linear_regression(train_data, test_partition=0):
    # features
    X = train_data.iloc[test_partition:, :-1]
    # target values
    y = train_data.iloc[test_partition:, -1]

    # Setting up the models
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    model = Model()
    model.fit(X, y, theta)
    return model

def logistic_regression(train_data, test_partition=0):
     # features
    X = train_data.iloc[test_partition:, :-1]
    # target values
    y = train_data.iloc[test_partition:, -1]

    # Setting up the models
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    model = LR_Model()
    model.train(X, y)
    return model

def tf_neural_network(train_data, out_size=1, test_partition=0):
    # features
    X = train_data.iloc[test_partition:, :-1]
    # target values
    y = train_data.iloc[test_partition:, -1]

    # Setting up the models
    # X = np.c_[np.ones((X.shape[0], 1)), X]
    X = X.to_numpy()
    # y = y[:, np.newaxis]
    y = y.to_numpy()
    model = TF_Model(in_size=X.shape[1], out_size=out_size)
    model.train(X, y)
    return model

if __name__ == "__main__":
    # create model for learning traffic based on timestamp
    traffic_data = c.load_data("Data/traffic.csv", 0).head(1000)
    traffic_data = c.clean_traffic_data(traffic_data)
    print(traffic_data.isnull().values.any())

    # tutorial: 
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
    # https://donaldpinckney.com/books/pytorch/book/ch2-linreg/2018-03-21-multi-variable.html