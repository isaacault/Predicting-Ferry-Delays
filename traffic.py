from __future__ import absolute_import, division, print_function, unicode_literals

import clean
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import pandas
import matplotlib.pyplot as plt


class Traffic:
    def __init__(self):
        self.train_dataset = clean.load_data("Data/traffic.csv", 0).head(1000)
        self.train_dataset = clean.clean_traffic_data(self.train_dataset)
        self.test_dataset = clean.load_data("Data/traffic.csv", 0).head(2000)[1000:]
        self.test_dataset = clean.clean_traffic_data(self.test_dataset)

    def regression(self):
        def build_model():
            model = keras.Sequential([
                  layers.Dense(64, activation='relu', input_shape=[len(self.train_dataset.keys())]),
                  layers.Dense(64, activation=None),
                  layers.Dense(1)
                  ])
            optimizer = tf.keras.optimizers.RMSprop(0.0001)
            model.compile(loss='mse',
                    optimizer=optimizer,
                    metrics=['mae', 'mse'])
            return model

        def norm(x):
            return (x - train_stats['mean']) / train_stats['std']

        def plot_history(history):
            hist = pandas.DataFrame(history.history)
            hist['epoch'] = history.epoch

            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Abs Error [Traffic.Ordinal]')
            plt.plot(hist['epoch'], hist['mae'],
                    label='Train Error')
            plt.plot(hist['epoch'], hist['val_mae'],
                    label = 'Val Error')
            plt.ylim([0,5])
            plt.legend()

            plt.figure()
            plt.xlabel('Epoch')
            plt.ylabel('Mean Square Error [$Traffic.Ordinal^2$]')
            plt.plot(hist['epoch'], hist['mse'],
                    label='Train Error')
            plt.plot(hist['epoch'], hist['val_mse'],
                    label = 'Val Error')
            plt.ylim([0,20])
            plt.legend()
            plt.show()
            
        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print('')
                print('.', end='')

        model = build_model()

        train_stats = self.train_dataset.describe()
        train_stats.pop("Traffic.Ordinal")
        train_stats = train_stats.transpose()
        normed_train_data = norm(self.train_dataset)
        normed_test_data = norm(self.test_dataset)
        train_labels = self.train_dataset.pop('Traffic.Ordinal')
        test_labels = self.test_dataset.pop('Traffic.Ordinal')
        
        EPOCHS = 1000
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
        print(normed_train_data)
        print(normed_train_data.shape)
        history = model.fit(
                normed_train_data, train_labels,
                epochs=EPOCHS, validation_split = 0.2, verbose=0,
                callbacks=[PrintDot()])

        test_predictions = model.predict(normed_test_data).flatten()
        print(test_predictions)

        loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)

        print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

        plt.scatter(test_labels, test_predictions)
        plt.xlabel('True Values [MPG]')
        plt.ylabel('Predictions [MPG]')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([0,plt.xlim()[1]])
        plt.ylim([0,plt.ylim()[1]])
        _ = plt.plot([-100, 100], [-100, 100])
        plt.show()


if __name__ == "__main__":
    train_dataset = Traffic()
    train_dataset.regression()

