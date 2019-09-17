import numpy as np

import clean as c
from train import Model

if __name__ == "__main__":
    data = c.load_data("Data/train.csv", 0)
    
    data = c.clean_trips(data)
    data = c.clean_date_time(data) 
    data = c.clean_status(data)
    data = c.clean_vessels(data)

    # features
    X_test = data.iloc[:10000, :-1]
    # target values
    y_test = data.iloc[:10000, -1]

    X = data.iloc[10000:, :-1]
    y = data.iloc[10000:, -1]
    print(X.size)

    # Setting up the models
    X = np.c_[np.ones((X.shape[0], 1)), X]
    y = y[:, np.newaxis]
    theta = np.zeros((X.shape[1], 1))
    model = Model()
    model.train(X, y)
    X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]
    y_test = y_test[:, np.newaxis]
    accuracy = model.test_accuracy(X_test, y_test)
    print(accuracy)

    # tutorial: 
    # https://towardsdatascience.com/building-a-logistic-regression-in-python-301d27367c24
    # def sigmoid(x):
    #     # Activation function used to map any real value between 0 and 1
    #     return 1 / (1 + np.exp(-x))

    # def net_input(theta, x):
    #     # Computes the weighted sum of inputs
    #     return np.dot(x, theta)

    # def probability(theta, x):
    #     # Returns the probability after passing through sigmoid
    #     return sigmoid(net_input(theta, x))


    # def cost_function(theta, x, y):
    #     # Computes the cost function for all the training samples
    #     m = x.shape[0]
    #     total_cost = -(1 / m) * np.sum(
    #         y * np.log(probability(theta, x)) + (1 - y) * np.log(
    #             1 - probability(theta, x)))
    #     return total_cost

    # def gradient(theta, x, y):
    #     # Computes the gradient of the cost function at the point theta
    #     m = x.shape[0]
    #     return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

    # def fit(x, y, theta):
    #     opt_weights = fmin_tnc(func=cost_function, x0=theta,
    #                         fprime=gradient,args=(x, y.flatten()))
    #     return opt_weights[0]

    # parameters = fit(X, y, theta)
    # print(parameters)

    # def predict(x):
    #     theta = parameters[:, np.newaxis]
    #     return probability(theta, x)
    
    # def accuracy(x, actual_classes, probab_threshold=0.5):
    #     predicted_classes = (predict(x) >= 
    #                         probab_threshold).astype(int)
    #     predicted_classes = predicted_classes.flatten()
    #     accuracy = np.mean(predicted_classes == actual_classes)
    #     return accuracy * 100

    # print(accuracy(X, y.flatten()))