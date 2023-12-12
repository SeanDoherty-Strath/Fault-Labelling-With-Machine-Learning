# The intention of this is to figure out why I can't get the model to overtrain.  Then it can be deleted
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import keras
from keras import layers
import pandas as pd
import pyreadr
import math
from pathlib import Path
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_absolute_error
from keras.optimizers import Adam


def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
all_df = rdata_read["faulty_training"]

n = 100
df = all_df.iloc[:n, 3:7]
# df = normalize_dataframe(df)


#

# Define input layer
input_layer = keras.Input(shape=(4,))

# Encoder
encoder = layers.Dense(5, activation='relu')(input_layer)
decoder = layers.Dense(4, activation='sigmoid')(encoder)

# Create autoencoder model
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Compile the model
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
optimizer = Adam(lr=100)
autoencoder.compile(optimizer=optimizer, loss='mse')

# Print the model summary
# autoencoder.summary()

#  prepare  input data.
xTrain = df.iloc[:math.floor(n*4/5), :]  # first 4/5 for training
xTest = df.iloc[math.ceil(n*4/5):, :]  # final 1/5  for testing


print('xTest size', np.shape(xTest))

losses = []
epochs = []

# Callback to record loss and epochs during training


class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epochs.append(epoch)
        losses.append(logs['loss'])


# Train the data: note - get more info on batchsize

autoencoder.fit(xTrain, xTrain, epochs=200,
                shuffle=True, validation_data=(xTest, xTest), callbacks=[LossHistory()])

plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()


# Encode and decode some data
predictedData = autoencoder.predict(xTest)
mae = mean_absolute_error(xTest, predictedData)  # mean absolute error
print('Mean absolte error: ', mae)


# Option 2: Time series
plt.figure(figsize=(20, 4))

for i in range(4):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(xTest.iloc[:, i])
    plt.ylim(0, 1)

    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(predictedData[:, i])
    plt.ylim(0, 1)
plt.show()
