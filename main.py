from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K

import clean

print(tf.__version__)

dataset = clean.load_data("Data/train.csv", 0)
dataset = clean.clean_trips(dataset)
dataset = clean.clean_date_time(dataset)
dataset = clean.clean_status(dataset)
dataset = clean.clean_vessels(dataset)
print(dataset.shape)
#dataset = clean.clean_traffic_data(dataset)
dataset = dataset.dropna()
for col in dataset.columns:
    dataset[col] = dataset[col].astype(int)
print(dataset)


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[["Delay.Indicator", "timestamp"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Delay.Indicator")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('Delay.Indicator')
test_labels = test_dataset.pop('Delay.Indicator')
print(train_stats['mean'])
print(train_stats['std'])
print(type(train_stats))

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_train_data)


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0005)
  #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=[auc])
  return model

EPOCHS = 15

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
print(model.summary())
example_batch = normed_train_data[:10]
print(example_batch)
print(train_labels[:10])
example_result = model.predict(example_batch)
print(example_result)

# The patience parameter is the amount of epochs to check for improvement
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split = 0.2, verbose=1)

#plot_history(history)

test_predictions = model.predict(normed_test_data).flatten()
print(test_predictions)
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Delay.Indicator]')
plt.ylabel('Predictions [Delay.Indicator]')
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