import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import keras
from keras import layers
import pandas as pd
import pyreadr
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

# Read in data
rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")

# Create a Pandas data frame
all_df = rdata_read["faulty_training"]

# Reduce data
n = 10000  # Number of time values
df = all_df.iloc[:n, 3:]

# Normalize data


def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


df = normalize_dataframe(df)

# Define the dimensions
input_output_dimension = 52
hidden_layer_dimension = 25
encoding_dimension = 12

# INPUT LAYER
input_layer = keras.Input(shape=(input_output_dimension,))
# input_layer = layers.Dropout(0.2)(input_layer)

# ENCODER
encoder = layers.Dense(hidden_layer_dimension, activation='relu',)(input_layer)
# encoder = layers.Dropout(0.2)(encoder)
encoder = layers.Dense(encoding_dimension, activation='relu')(encoder)

# DECODER
decoder = layers.Dense(25, activation='relu')(encoder)
# decoder = layers.Dropout(0.2)(decoder)
decoder = layers.Dense(input_output_dimension, activation='sigmoid')(decoder)

# AUTOENCODER
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# COMPILE MODEL
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

# K FOLD VALIDATION
numFolds = 2
kf = KFold(n_splits=numFolds, shuffle=True, random_state=42)
# store performance metrics for each fold
fold_metrics = []

# Record losses and epochs during training
epochs = []
losses = []


class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epochs.append(epoch)
        losses.append(logs['loss'])


plt.figure(figsize=(20, 4))
for fold, (train_index, val_index) in enumerate(kf.split(df)):
    print(f"Training on fold {fold + 1}/{numFolds}")

    xTrain, xTest = df.iloc[train_index], df.iloc[val_index]

    # Train the autoencoder model on the current fold
    history = autoencoder.fit(
        xTrain, xTrain, epochs=1000, validation_data=(xTest, xTest), verbose=2, callbacks=[LossHistory()])

    ax = plt.subplot(1, numFolds, fold+1)
    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    epochs = []
    losses = []

    # Evaluate the model on the validation set and store the metrics
    val_loss = autoencoder.evaluate(xTest, xTest, verbose=0)
    fold_metrics.append(val_loss)

average_loss = np.mean(fold_metrics)
print(f"Average validation loss across {numFolds} folds: {average_loss}")
plt.show()


# VISUAL INPUT AGAINST RECREATION
predictedData = autoencoder.predict(df)
mae = mean_absolute_error(df, predictedData)  # mean absolute error
print('Mean absolte error: ', mae)

# Option 2: Time series
plt.figure(figsize=(20, 4))

for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(df.iloc[:, i])
    plt.ylim(0, 1)

    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(predictedData[:, i])
    plt.ylim(0, 1)
plt.show()
