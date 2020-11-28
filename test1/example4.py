import pathlib

import matplotlib.pyplot as plt
import pandas as pd

# 使用seaborn绘制矩阵图
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers

import tensorflow_docs as tfdocs
import tensorflow_docs.plots
import tensorflow_docs.modeling

dateset_path = r'C:/Users/Herrrb/Desktop/tf-examples/test1/auto-mpg.data'

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']
raw_dataset = pd.read_csv(dateset_path, names=column_names, na_values="?",
                          comment='\t', sep=' ', skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

dataset = dataset.dropna()

# origin = dataset.pop("Origin")
# dataset['USA'] = (origin == 1) * 1.0
# dataset['Europe'] = (origin == 2) * 1.0
# dataset['Japan'] = (origin == 3) * 1.0
dataset['Origin'] = dataset['Origin'].map({1: "USA", 2: "Europe", 3: "Japan"})
dataset = pd.get_dummies(dataset, prefix='', prefix_sep=' ')

print('=================================')
print(dataset.tail())


train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# sns.pairplot(train_dataset[["MPG", "Cylinders", "Displacement", "Weight"]], diag_kind="kde")
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

train_labels = train_dataset.pop("MPG")
test_labels = test_dataset.pop("MPG")


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


def build_model():
    model_ = tf.keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])
    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model_.compile(loss='mse',
                   optimizer=optimizer,
                   metrics=['mae', 'mse'])
    return model_


model = build_model()
Epochs = 1000
history = model.fit(normed_train_data, train_labels, epochs=Epochs,
                    validation_split=0.2, verbose=0,
                    callbacks=[tfdocs.modeling.EpochDots()])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())

# plot
plotter = tfdocs.plots.HistoryPlotter(smoothing_std=2)
plotter.plot({'Basic': history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

plotter.plot({'Basic': history}, metric="mse")
plt.ylim([0, 20])
plt.ylabel('MSE [MPG^2]')
plt.show()


model = build_model()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
early_history = model.fit(normed_train_data, train_labels,
                          epochs=Epochs, validation_split=0.2, verbose=0,
                          callbacks=[early_stop, tfdocs.modeling.EpochDots()])

plotter.plot({'Early Stopping': early_history}, metric="mae")
plt.ylim([0, 10])
plt.ylabel('MAE [MPG]')
plt.show()

loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
print("Test set Mean Abs Error: {:5.2f} MPG".format(mae))

test_predictions = model.predict(normed_test_data).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
lims = [0, 50]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
plt.show()

plt.figure(5)
plt.hist(train_labels, bins=25)
plt.xlabel("Distribution of MPG")
plt.ylabel("Count")
plt.show()
