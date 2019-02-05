from __future__ import absolute_import, division, print_function
import pathlib
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt


def norm(x):
    """
    Normalize the training data
    """
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    """
    Build the model
    """
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[
                     len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse']
                  )
    return model


print(tf.__version__)

# Download the data
dataset_path = keras.utils.get_file(
    "auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
)

print(dataset_path)

column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                'Acceleration', 'Model Year', 'Origin']

# Read the data
raw_dataset = pd.read_csv(
    dataset_path, names=column_names,
    na_values="?", comment='\t',
    sep=" ", skipinitialspace=True
)

dataset = raw_dataset.copy()
dataset.tail()

# Drop the NULL rows
dataset = dataset.dropna()

print(dataset)

# Convert "Origin to a one hot" since Origin is categorical
origin = dataset.pop('Origin')

dataset['USA'] = (origin == 1)*1.0
dataset['Europe'] = (origin == 2)*1.0
dataset['Japan'] = (origin == 3)*1.0
dataset.tail()

# Split the training set and test set
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

# Describe the data
train_stats = train_dataset.describe()
train_stats.pop("MPG")
train_stats = train_stats.transpose()

# Seperate the dependent vars from independent vars
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')

# Normalize the data
normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)

# Build the model
model = build_model()

# Print the model Summary
model.summary()


# Train the model for 1000 epochs
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:
            print('')
        print('.', end='')


EPOCHS = 1000

# Fitting our model to the neural net
history = model.fit(
    normed_train_data, train_labels,
    epochs=EPOCHS, validation_split=0.2, verbose=0,
    callbacks=[PrintDot()])


# Time to make prediction
print(normed_test_data)
test_predictions = model.predict(normed_test_data).flatten()

# Plot the regression result on a graph
plt.scatter(test_labels, test_predictions)
plt.xlabel('True Values [MPG]')
plt.ylabel('Predictions [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()
