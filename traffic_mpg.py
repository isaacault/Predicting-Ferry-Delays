from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

import clean

print(tf.__version__)

dataset = clean.load_data("Data/traffic.csv", 0).head(600000)
print(dataset.shape)
#dataset = clean.clean_traffic_data(dataset)
dataset = dataset.dropna()
#for col in dataset.columns:
#    dataset[col] = dataset[col].astype(int)
print(dataset)


train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

#sns.pairplot(train_dataset[["Traffic.Ordinal", "timestamp"]], diag_kind="kde")

train_stats = train_dataset.describe()
train_stats.pop("Traffic.Ordinal")
train_stats = train_stats.transpose()
print(train_stats)

train_labels = train_dataset.pop('Traffic.Ordinal')
test_labels = test_dataset.pop('Traffic.Ordinal')
print(train_stats['mean'])
print(train_stats['std'])
train_stats['std']['Year'] = 1
train_stats['std']['Month'] = 1
train_stats['std']['Day'] = 1
print(type(train_stats))

def norm(x):
  return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)
print(normed_train_data)


def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)
  #optimizer = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

  model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])
  return model


# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 100 == 0: print('')
    print('.', end='')

EPOCHS = 1000

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Traffic.Ordinal]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Traffic.Ordinal^2$]')
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
                    validation_split = 0.2, verbose=0, callbacks=[early_stop, PrintDot()])

plot_history(history)

test_predictions = model.predict(normed_test_data).flatten()
print(test_predictions)
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [Traffic.Ordinal]')
plt.ylabel('Predictions [Traffic.Ordinal]')
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
#plt.show()