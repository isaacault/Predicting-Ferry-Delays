from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
import keras

import numpy as np
import pandas as pd
from scipy.optimize import fmin_tnc, check_grad, least_squares

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 


class LR_Model():
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = LogisticRegression(solver='liblinear')

    def train(self, X, y):    
        self.model.fit(X, y.ravel())
    
    def accuracy(self, x, actual_classes):
        predicted_classes = self.model.predict(x)
        accuracy = accuracy_score(actual_classes.flatten(), predicted_classes)
        print(accuracy)
        predicted_probs = self.model.predict_proba(x)
        predicted_probs = predicted_probs.flatten()
        print(predicted_probs)
        return auc(predicted_classes, actual_classes)

# Some utility functions used for computing in the model
def sigmoid(x):
    # Activation function used to map any real value between 0 and 1
    return 1 / (1 + np.exp(-x))

def net_input(theta, x):
    # Computes the weighted sum of inputs
    return np.dot(x, theta)

def probability(theta, x):
    # Returns the probability after passing through sigmoid
    return sigmoid(net_input(theta, x))

def cost_function(theta, x, y):
    # Computes the cost function for all the training samples
    m = x.shape[0]
    total_cost = -(1 / m) * np.sum(y * np.log(probability(theta, x)) + (1 - y) * np.log(1 - probability(theta, x)))
    return total_cost

class Model():

    def __init__(self, *args, **kwargs):
        return

    def gradient(self, theta, x, y):
        # Computes the gradient of the cost function at the point theta
        m = x.shape[0]
        return (1 / m) * np.dot(x.T, sigmoid(net_input(theta,   x)) - y)

    def fit(self, x, y, theta):
        # minimizes cost function
        print(theta)
        opt_weights = fmin_tnc(func=cost_function, x0=theta,
                                fprime=self.gradient, args=(x, y.flatten()))
        self.params = opt_weights[0]

    def predict(self, x):
        theta = self.params[:, np.newaxis]
        return probability(theta, x)

    def accuracy(self, x, actual_classes):
        predicted_classes = self.predict(x)
        predicted_classes = predicted_classes.flatten()
        return auc(predicted_classes, actual_classes)

class TF_Model():
    def __init__(self, in_size, out_size, *args, **kwargs):
        self.model = keras.Sequential([
            keras.layers.Dense(8, activation='relu', input_dim=in_size, kernel_initializer='random_normal'),
            # keras.layers.Dense(4, activation=tf.nn.relu),
            keras.layers.Dense(1, activation='sigmoid')
        ])

        metrics = [
                keras.metrics.Accuracy(name='accuracy'),
                keras.metrics.TruePositives(name='tp'),
                keras.metrics.FalsePositives(name='fp'),
                keras.metrics.TrueNegatives(name='tn'),
                keras.metrics.FalseNegatives(name='fn'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc')
            ]

        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=metrics)

    def train(self, X, y):
        self.model.fit(X, y, epochs=3)

    def accuracy(self, X, actual_classes):
        predicted_classes = self.model.predict(X)
        print(predicted_classes)
        predicted_classes = predicted_classes.flatten()
        return auc(predicted_classes, actual_classes)

def auc(predicted_classes, actual_classes):
        fpr, tpr, thresholds = metrics.roc_curve(actual_classes, predicted_classes, pos_label=1)
        accuracy = metrics.auc(fpr, tpr)
        return accuracy

def auc_alt(y_true, y_pred):
    try:
        return tf.py_func(metrics.roc_auc_score, (y_true, y_pred), tf.double)
    except:
        pass
