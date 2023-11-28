import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import keras
from keras import layers
import pandas as pd
import pyreadr
import math
from pathlib import Path

dataSize = 10000  # The amount of time values used


def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


df = pd.read_csv(
    "TenesseeEastemen_FaultyTraining_Subsection.csv")
df = df.iloc[:, 3:]

rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
all_df = rdata_read["faulty_training"]
df = all_df.iloc[:dataSize, 3:]


df = normalize_dataframe(df)


# Define the dimensions
input_output_dimension = 52
encoding_dimension = 12

# Define input layer
input_layer = keras.Input(shape=(input_output_dimension,))

# Encoder
encoder = layers.Dense(25, activation='relu')(input_layer)
encoder = layers.Dense(encoding_dimension, activation='relu')(encoder)

# Decoder
decoder = layers.Dense(25, activation='relu')(encoder)
decoder = layers.Dense(input_output_dimension, activation='sigmoid')(decoder)

# Create the autoencoder model
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Compile the model
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

# Print the model summary
# autoencoder.summary()


#  prepare  input data.
xTrain = df.iloc[:math.floor(dataSize*2/3), :]  # first 2 thirds for training
xTest = df.iloc[math.ceil(dataSize*2/3):, :]  # final third for testing

print('xTest size', np.shape(xTest))

losses = []
epochs = []

# Callback to record loss and epochs during training


class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epochs.append(epoch)
        losses.append(logs['loss'])


# Train the data: note - get more info on batchsize
autoencoder.fit(xTrain, xTrain, epochs=50,
                shuffle=True, validation_data=(xTest, xTest), callbacks=[LossHistory()])

plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()


# Encode and decode some data
predictedData = autoencoder.predict(xTest)


# DISPLAY RESULTS
# Option 1: 'Snapshots'
n = 5  # How many graphs to display
plt.figure(figsize=(20, 4))

# for i in range(n):
#     # Display original
#     ax = plt.subplot(2, n, i + 1)
#     plt.stem(xTest.iloc[i, :])

#     # Display reconstruction
#     ax = plt.subplot(2, n, i + 1 + n)
#     plt.stem(predictedData[i])
# plt.show()

# Option 2: Time series
plt.figure(figsize=(20, 4))

for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(xTest.iloc[:, i])
    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(predictedData[:, i])
plt.show()


# encoder = keras.Model(inputs=autoencoder.input,
#     outputs=autoencoder.get_layer('encoder_1').output)
encoded = keras.Model(inputs=autoencoder.input, outputs=encoder)

# Get the latent space representation for the input data
latent_space = encoded.predict(df)

print('Latent space', latent_space)
print('Latent space size: ', np.shape(latent_space))


# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot the points in the latent space
ax.scatter(latent_space[:, 3], latent_space[:, 4],
           latent_space[:, 5], marker='o', s=10, c='r')


for i in range(0, encoding_dimension):
    text = 'Latent space', i
    print(text, latent_space[:, i])


latentSpaceDF = pd.DataFrame(latent_space)
filepath = Path('./LatentSpace.csv')

latentSpaceDF.to_csv(filepath)

ax.set_xlabel('Latent Dimension 1')
ax.set_ylabel('Latent Dimension 2')
ax.set_zlabel('Latent Dimension 3')

plt.title('Latent Space Visualization')
plt.show()
