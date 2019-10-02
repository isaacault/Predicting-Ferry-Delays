from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib
import matplotlib.pyplot as plt
import pandas as pd
#import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import keras.backend as K
import load_data
from collections import OrderedDict

EPOCHS = 15

def auc(y_true, y_pred):
    auc = tf.metrics.auc(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

def build_model():
  model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=[len(train_data.keys())]),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
  ])

  optimizer = tf.keras.optimizers.RMSprop(0.0001)

  model.compile(loss='binary_crossentropy',
                optimizer=optimizer,
                metrics=[auc])
  return model

def norm(x):
    return (x - train_stats['mean']) / train_stats['std']



train_data, test_data = load_data.get_data()
train_data = train_data.sample(frac=1)
train_labels = train_data.pop('Delay.Indicator')
train_stats = train_data.describe()
train_stats = train_stats.transpose()
submission_id = test_data.pop('ID')

model = build_model()

normed_train_data = norm(train_data)
normed_test_data = norm(test_data)

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(
  normed_train_data, train_labels,
  epochs=EPOCHS, verbose=1)

test_predictions = model.predict(normed_test_data).flatten()

#test_predictions['ID'] = test_predictions.index

print(test_predictions)

#submission_data = pd.DataFrame({'ID': submission_id, 'Delay.Inidcator':test_predictions})
submission_data = pd.DataFrame(
  OrderedDict(
    {
      'ID' : pd.Series(submission_id),
      'Delay.Indicator' : pd.Series(test_predictions)
    }
  )
)
print(submission_data)
submission_data.to_csv('submission.csv', index=False)