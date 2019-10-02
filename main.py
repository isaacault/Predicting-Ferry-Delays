from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import os
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
from sklearn import metrics

import clean

print(tf.__version__)
traffic_dataset = clean.load_data("Data/traffic.csv", 0)
vancouver_dataset = clean.load_data("Data/vancouver.csv", 0)
victoria_dataset = clean.load_data("Data/victoria.csv", 0)

if(os.path.isfile("Data/clean_dataset.csv")):
    dataset = clean.load_data("Data/clean_dataset.csv", 0)
else:
    dataset = clean.load_data("Data/train.csv", 0)
    dataset = dataset.sample(frac=1)
    dataset = clean.clean_trips(dataset)
    dataset = clean.clean_date_time(dataset)
    dataset = clean.clean_status(dataset)
    dataset = clean.clean_vessels(dataset)
    dataset = clean.stitch_traffic(dataset, traffic_dataset)
    dataset = clean.stitch_weather(dataset, vancouver_dataset, "vancouver")
    dataset = clean.stitch_weather(dataset, victoria_dataset, "victoria")
    dataset.to_csv("Data/clean_dataset.csv")

if(os.path.isfile("Data/clean_test.csv")):
    submission_dataset = clean.load_data("Data/clean_test.csv")    
else:
    submission_dataset = clean.load_data("Data/test.csv")
    submission_dataset = clean.clean_date_time(submission_dataset)
    submission_dataset = clean.clean_trips(submission_dataset)
    submission_dataset = clean.clean_vessels(submission_dataset)
    submission_dataset = clean.stitch_traffic(submission_dataset, traffic_dataset)
    submission_dataset = clean.stitch_weather(submission_dataset, vancouver_dataset, "vancouver")
    submission_dataset = clean.stitch_weather(submission_dataset, victoria_dataset, "victoria")
    submission_dataset.to_csv("Data/clean_test.csv")


dataset = dataset.dropna()
print(dataset.shape)
#for col in dataset.columns:
#    dataset[col] = dataset[col].astype(int)
EPOCHS = 15

train_dataset = dataset.iloc[10000:, :]
test_dataset = dataset.iloc[:10000, :]

submission_ids = submission_dataset.pop('ID')

#sns.pairplot(train_dataset[["Delay.Indicator", "timestamp"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Delay.Indicator")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop('Delay.Indicator')
test_labels = test_dataset.pop('Delay.Indicator')


def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def auc(y_true, y_pred):
    auc = tf.compat.v1.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
normed_submission_data = norm(submission_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)
  #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=[auc, ])
  return model

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Delay.Indicator]')

  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Delay.Indicator^2$]')

  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()

model = build_model()

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS, validation_split = 0.2, verbose=1)

test_predictions = model.predict(normed_test_data).flatten()
submission_predictions = model.predict(normed_submission_data).flatten()
fpr, tpr, thresholds = metrics.roc_curve(test_labels, test_predictions, pos_label=1)
accuracy = metrics.auc(fpr, tpr)
print(accuracy)
print(pd.DataFrame({'ID': [submission_ids], 'Delay.Indicator': [submission_predictions]}))

plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Delay_Indicator]')
plt.ylabel('Predictions [Delay_Indicator]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0,plt.xlim()[1]])
plt.ylim([0,plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")

plt.show()

