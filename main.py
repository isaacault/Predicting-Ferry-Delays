import numpy as np

import clean as c
# from train import Model
from model import Model, TF_Model

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

    data = c.load_data("Data/train.csv", 0)
    test_data = c.load_data("Data/test.csv", 0)
    test_partition = 10000 # for TEST

    train_data = data
    train_data = c.clean_trips(train_data)
    train_data = c.clean_date_time(train_data) 
    train_data = c.clean_status(train_data)
    train_data = c.clean_vessels(train_data)

    test_data = c.clean_trips(test_data, False)
    test_data = c.clean_date_time(test_data)
    test_data = c.clean_vessels(test_data)

    model = tf_neural_network(train_data, test_partition)
    # TEST
    # model = linear_regression(train_data, test_partition)
    X_test = train_data.iloc[:test_partition, :-1]
    # # target values
    y_test = train_data.iloc[:test_partition, -1]
    # X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    X_test = X_test.to_numpy()
    # y_test = y_test[:, np.newaxis]
    y_test = y_test.to_numpy()
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