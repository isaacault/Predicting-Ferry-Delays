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

def tf_neural_network(train_data, test_partition=0):
    # features
    X = train_data.iloc[test_partition:, :-1]
    # target values
    y = train_data.iloc[test_partition:, -1]

    # Setting up the models
    # X = np.c_[np.ones((X.shape[0], 1)), X]
    X = X.to_numpy()
    # y = y[:, np.newaxis]
    y = y.to_numpy()
    model = TF_Model()
    print(model.accuracy(X, y))
    model.train(X, y)
    return model

if __name__ == "__main__":
    test_partition = 10000 # for TEST    
    data = c.load_data("Data/train.csv", 0)
    traffic_data = c.load_data("Data/traffic.csv", 0)
    vancouver_data = c.load_data("Data/vancouver.csv", 0)
    victoria_data = c.load_data("Data/victoria.csv", 0)
    train_data = shuffle(data)
    train_data = c.clean_trips(train_data)
    train_data = c.clean_date_time(train_data) 
    train_data = c.clean_status(train_data)
    train_data = c.clean_vessels(train_data)
    train_data.to_csv("Data/clean_train.csv")    
    print(train_data)
    model = linear_regression(train_data, test_partition)
    # model = logistic_regression(train_data)

    # TEST
    # model = linear_regression(train_data, test_partition)
    X_test = train_data.iloc[:test_partition, :-1]
    # # target values
    y_test = train_data.iloc[:test_partition, -1]

    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    y_test = y_test[:, np.newaxis]
    accuracy = model.accuracy(X_test, y_test)
    print(accuracy)

    # # REAL
    # model = linear_regression(train_data)
    # X_test = test_data.iloc[:, 1:]
    # ids = test_data.iloc[:, 1]
    # X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    # predictions = model.predict(X_test)

    # tutorial: 
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
    # https://donaldpinckney.com/books/pytorch/book/ch2-linreg/2018-03-21-multi-variable.html