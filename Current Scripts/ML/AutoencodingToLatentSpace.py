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


def normalize_dataframe(df):
    normalized_df = pd.DataFrame()
    for col in df.columns:  # Ignore the first four columns
        if df.columns.get_loc(col) < 4:
            normalized_df[col] = df[col]
        else:  # Noramlize everything else between 0 and 1
            normalized_col = (df[col] - df[col].min()) / \
                (df[col].max() - df[col].min())
            normalized_df[col] = normalized_col

    return normalized_df


# Read in data
all_df = pd.read_csv(
    "Data/UpdatedData.csv")
all_df = normalize_dataframe(all_df)
df = all_df.iloc[:, 4:]
print(df.shape)


# Define the dimensions
input_output_dimension = 52
hidden_layer_dimension = 25
encoding_dimension = 12

# INPUT LAYER
input_layer = keras.Input(shape=(input_output_dimension,))
input_layer = layers.Dropout(0.1)(input_layer)

# ENCODER
encoder = layers.Dense(hidden_layer_dimension, activation='relu',)(input_layer)
encoder = layers.Dropout(0.1)(encoder)
encoder = layers.Dense(encoding_dimension, activation='relu')(encoder)

# DECODER

decoder = layers.Dense(hidden_layer_dimension, activation='relu')(encoder)
# decoder = layers.Dropout(0.2)(decoder)
decoder = layers.Dense(input_output_dimension, activation='sigmoid')(decoder)

# AUTOENCODER
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# COMPILE MODEL
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')

# Print the model summary
# autoencoder.summary()

#  prepare  input data.
xTrain = df.iloc[:16000, :]  # first 4/5 for training
xTest = df.iloc[16000:, :]  # final 1/5  for testing


losses = []
epochs = []

# Callback to record loss and epochs during training


class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epochs.append(epoch)
        losses.append(logs['loss'])


# Train the data: note - get more info on batchsize
autoencoder.fit(xTrain, xTrain, epochs=100,
                shuffle=True, validation_data=(xTest, xTest), callbacks=[LossHistory()])

plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()


# TEST THE DATA ON THE TEST SET
testDF = pd.read_csv(
    "Data/UpdatedTestData.csv")
testDF = normalize_dataframe(testDF)
testDF = testDF.iloc[:, 4:]

predictedData = autoencoder.predict(testDF)
mae = mean_absolute_error(testDF, predictedData)  # mean absolute error
print('Mean absolte error: ', mae)

# Option 2: Time series
plt.figure(figsize=(20, 4))

for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(testDF.iloc[:960, i])
    plt.ylim(0, 1)

    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(predictedData[:960, i])
    plt.ylim(0, 1)
plt.show()


# CONVERT TO LATENT SPACE
# # encoder = keras.Model(inputs=autoencoder.input,
# #     outputs=autoencoder.get_layer('encoder_1').output)
# encoded = keras.Model(inputs=autoencoder.input, outputs=encoder)

# # Get the latent space representation for the input data
# latent_space = encoded.predict(df)

# print('Latent space', latent_space)
# print('Latent space size: ', np.shape(latent_space))


# # Create a 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')

# # Scatter plot the points in the latent space
# # ax.scatter(latent_space[:, 3], latent_space[:, 4],
# #           latent_space[:, 5], marker='o', s=10, c='r')


# for i in range(0, encoding_dimension):
#     text = 'Latent space', i
#     print(text, latent_space[:, i])


# latentSpaceDF = pd.DataFrame(latent_space)
# filepath = Path('./LatentSpace.csv')

# latentSpaceDF.to_csv(filepath)

# ax.set_xlabel('Latent Dimension 1')
# ax.set_ylabel('Latent Dimension 2')
# ax.set_zlabel('Latent Dimension 3')

# plt.title('Latent Space Visualization')
# plt.show()
