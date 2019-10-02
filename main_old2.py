#!/usr/bin/env python3

from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from pathlib import Path

print(tf.__version__)

train_data = Path("Data/train.csv")
column_names = [
        'Vessel_Name',
        'Scheduled_Departure',
        'Status',
        'Trip',
        'Trip_Duration',
        'Day',
        'Month',
        'Day_of_Month',
        'Year',
        'Full_Date',
        'Delay_Indicator'
    ]

raw_dataset = pd.read_csv(train_data, names=column_names, sep=",", skipinitialspace=True)
dataset = raw_dataset.copy()
dataset = dataset.dropna()
dataset.pop('Status')
dataset.pop('Full_Date')
dataset = dataset.iloc[1:]
vessel_name = dataset.pop('Vessel_Name')
trip = dataset.pop('Trip')


dataset['Spirit of British Columbia'] = (vessel_name == 'Spirit of British Columbia')*1.0
dataset['Queen of New Westminster'] = (vessel_name == 'Queen of New Westminster')*1.0
dataset['Spirit of Vancouver Island'] = (vessel_name == 'Spirit of Vancouver Island')*1.0
dataset['Coastal Celebration'] = (vessel_name == 'Coastal Celebration')*1.0
dataset['Queen of Alberni'] = (vessel_name == 'Queen of Alberni')*1.0
dataset['Coastal Inspiration'] = (vessel_name == 'Coastal Inspiration')*1.0
dataset['Skeena Queen'] = (vessel_name == 'Skeena Queen')*1.0
dataset['Coastal Renaissance'] = (vessel_name == 'Coastal Renaissance')*1.0
dataset['Queen of Oak Bay'] = (vessel_name == 'Queen of Oak Bay')*1.0
dataset['Queen of Cowichan'] = (vessel_name == 'Queen of Cowichan')*1.0
dataset['Queen of Capilano'] = (vessel_name == 'Queen of Capilano')*1.0
dataset['Queen of Surrey'] = (vessel_name == 'Queen of Surrey')*1.0
dataset['Queen of Coquitlam'] = (vessel_name == 'Queen of Coquitlam')*1.0
dataset['Bowen Queen'] = (vessel_name == 'Bowen Queen')*1.0
dataset['Queen of Cumberland'] = (vessel_name == 'Queen of Cumberland')*1.0
dataset['Island Sky'] = (vessel_name == 'Island Sky')*1.0
dataset['Mayne Queen'] = (vessel_name == 'Mayne Queen')*1.0

dataset['Tsawwassen to Swartz Bay'] = (trip == 'Tsawwassen to Swartz Bay')*1.0
dataset['Tsawwassen to Duke Point'] = (trip == 'Tsawwassen to Duke Point')*1.0
dataset['Swartz Bay to Fulford Harbour (Saltspring Is.)'] = (trip == 'Swartz Bay to Fulford Harbour (Saltspring Is.)')*1.0
dataset['Swartz Bay to Tsawwassen'] = (trip == 'Swartz Bay to Tsawwassen')*1.0
dataset['Duke Point to Tsawwassen'] = (trip == 'Duke Point to Tsawwassen')*1.0
dataset['Departure Bay to Horseshoe Bay'] = (trip == 'Departure Bay to Horseshoe Bay')*1.0
dataset['Horseshoe Bay to Snug Cove (Bowen Is.)'] = (trip == 'Horseshoe Bay to Snug Cove (Bowen Is.)')*1.0
dataset['Horseshoe Bay to Departure Bay'] = (trip == 'Horseshoe Bay to Departure Bay')*1.0
dataset['Horseshoe Bay to Langdale'] = (trip == 'Horseshoe Bay to Langdale')*1.0
dataset['Langdale to Horseshoe Bay'] = (trip == 'Langdale to Horseshoe Bay')*1.0

def weekdayToInt(weekday):
    return {
        'Monday': 0,
        'Tuesday': 1,
        'Wednesday': 2,
        'Thursday': 3,
        'Friday': 4,
        'Saturday': 5,
        'Sunday': 6
    }[weekday]

def monthToInt(month):
    return {
        'January': 0,
        'February': 1,
        'March': 2,
        'April': 3,
        'May': 4,
        'June': 5,
        'July': 6,
		'August' : 7,
		'September' : 8,
		'October' : 9,
		'November' : 10,
		'December' : 11
    }[month]

def toTwentyFour(time):
    temp_list = time.split(" ")
    return int(temp_list[0].replace(':', '')) if temp_list[1] == "AM" else int(temp_list[0].replace(':', '')) + 1200

dataset['Scheduled_Departure'] = dataset['Scheduled_Departure'].apply(toTwentyFour)
dataset['Day'] = dataset['Day'].apply(weekdayToInt)
dataset['Month'] = dataset['Month'].apply(monthToInt)

for col in dataset.columns:
    dataset[col] = dataset[col].astype(int)

train_dataset = dataset.sample(frac=0.8,random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_labels = train_dataset.pop('Delay_Indicator')
test_labels = test_dataset.pop('Delay_Indicator')

train_stats = train_dataset.describe()
train_stats = train_stats.transpose()

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']

normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(1)
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.001)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=['mean_absolute_error', 'mean_squared_error'])
  return model

model = build_model()

print(model.summary())

EPOCHS = 3

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, validation_split = 0.2, verbose=1,
  callbacks=[early_stop])

def plot_history(history):
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Abs Error [Delay_Indicator]')
  plt.plot(hist['epoch'], hist['mean_absolute_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
           label = 'Val Error')
  plt.ylim([0,5])
  plt.legend()

  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Square Error [$Delay_Indicator^2$]')
  plt.plot(hist['epoch'], hist['mean_squared_error'],
           label='Train Error')
  plt.plot(hist['epoch'], hist['val_mean_squared_error'],
           label = 'Val Error')
  plt.ylim([0,20])
  plt.legend()
  plt.show()


#plot_history(history)

test_predictions = model.predict(normed_test_data).flatten()

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
#plt.show()