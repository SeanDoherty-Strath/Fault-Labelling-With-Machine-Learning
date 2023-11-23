import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import keras
from keras import layers
import pandas as pd
import pyreadr


def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


df = pd.read_csv(
    "TenesseeEastemen_FaultyTraining_Subsection.csv")
df = df.iloc[:, 3:]

rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
all_df = rdata_read["faulty_training"]
df = all_df.iloc[:1000, 3:]

print(df.shape)

df = normalize_dataframe(df)

encodingDimension = 12  # We're having a bottleneck of four neurons
inputOutputDimension = 52  # Our input and output is seven sensors

input = keras.Input(shape=(inputOutputDimension,))

# encoded representation of the input
encoded = layers.Dense(encodingDimension, activation='relu')(input)

#  lossy reconstruction of the input
decoded = layers.Dense(inputOutputDimension, activation='sigmoid')(
    encoded)  # change to relu?


#  maps an input to its encoded representation
encoder = keras.Model(input, encoded)

#  maps an input to its reconstruction
autoencoder = keras.Model(input, decoded)

#  encoded input (i.e. small dimensions)
encodedInput = keras.Input(shape=(encodingDimension,))

# get  last layer  autoencoder model
decoderLayer = autoencoder.layers[-1]

# Create the decoder model
decoder = keras.Model(encodedInput, decoderLayer(encodedInput))

# Now train autoencoder to reconstruct sampled data

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

#  prepare  input data.
xTrain = df.iloc[:200, :]  # first 2 thirds for training
xTest = df.iloc[200:, :]  # final third for testing


print(xTrain.shape)
print(xTest.shape)


# Train the data: note - get more info on batchsize
autoencoder.fit(xTrain, xTrain, epochs=50, batch_size=inputOutputDimension,
                shuffle=True, validation_data=(xTest, xTest))

# Encode and decode some data
encodedData = encoder.predict(xTest)
decodedData = decoder.predict(encodedData)


# DISPLAY RESULTS
# Option 1: 'Snapshots'
n = 5  # How many graphs to display
plt.figure(figsize=(20, 4))

for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.stem(xTest.iloc[i, :])

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.stem(decodedData[i])
plt.show()

# Option 2: Time series
plt.figure(figsize=(20, 4))
print('Decoded dimensions', decodedData[:, 1])

for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(xTest.iloc[:, i])
    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(decodedData[:, i])
plt.show()
