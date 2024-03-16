import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
import pandas as pd
from sklearn.metrics import mean_absolute_error
import time
import tensorflow as tf
import tensorflow_addons as tfa


def performAutoEncoding(data):
    start_time = time.time()
    # Standardize data
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    # Create new df with the normalized values
    data = pd.DataFrame(normalized_values, columns=data.columns)

    # Add noise to input data

    noise = np.random.normal(0.0, scale=1.0, size=data.shape)

    # data2 = data + 0 * noise

    # return data
    # length = data.shape[1]

    # # Define the dimensions
    # input_output_dimension = 52
    # # hidden_layer_dimension =
    # encoding_dimension = 13

    # # INPUT LAYER
    # input_layer = keras.Input(shape=(52,))
    # # input_layer = layers.Dropout(0.01)(input_layer)
    # encoder = layers.Dense(
    #     32, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.0005))(input_layer)
    # # encoder = layers.Dropout(0.01)(encoder)
    # hidden_layer = layers.Dense(
    #     16, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.0005))(encoder)
    # decoder = layers.Dense(32, activation='relu')(hidden_layer)
    # output_layer = layers.Dense(52, activation='linear')(decoder)

    input_layer = keras.Input(shape=(52,))
    # input_layer = layers.Dropout(0.01)(input_layer)
    encoder = layers.Dense(
        32, activation='relu')(input_layer)
    # encoder = layers.Dropout(0.01)(encoder)
    hidden_layer = layers.Dense(
        16, activation='relu', )(encoder)
    decoder = layers.Dense(32, activation='relu')(hidden_layer)
    output_layer = layers.Dense(52, activation='linear')(decoder)

    # # AUTOENCODER

    #     # Choose regularization type (L1 or L2) and rate
    # regularization_type = 'l1'  # Change to 'l2' if you want L2 regularization
    # regularization_rate = 0.001

    # # Define and compile the autoencoder model with regularization
    # regularization = None
    # if regularization_type == 'l1':
    #     regularization = tf.keras.regularizers.l1
    # elif regularization_type == 'l2':
    #     regularization = tf.keras.regularizers.l2

    # autoencoder = autoencoder_with_regularization(input_shape, latent_dim, regularization, regularization_rate)
    # autoencoder.compile(optimizer='adam', loss='mse')

    # def autoencoder_with_regularization(regularization, regularization_rate=0.01):

    #     # input_layer = keras.Input(shape=(52,))
    #     # encoder = layers.Dense(32, activation='relu', activity_regularizer=regularization(
    #     #     regularization_rate))(input_layer)
    #     # hidden_layer = layers.Dense(
    #     #     8, activation='relu', activity_regularizer=regularization(regularization_rate))(encoder)
    #     # decoder = layers.Dense(32, activation='relu',)(hidden_layer)
    #     # output_layer = layers.Dense(52, activation='linear')(decoder)

    #     input_layer = keras.Input(shape=(52,))
    #     encoder = layers.Dense(32, activation='relu', )(input_layer)
    #     hidden_layer = layers.Dense(8, activation='relu')(encoder)
    #     decoder = layers.Dense(32, activation='relu',)(hidden_layer)
    #     output_layer = layers.Dense(52, activation='linear')(decoder)

    #     autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

    #     return autoencoder

    # regularization = tf.keras.regularizers.l2
    # regularization_rate = 0.01

    # autoencoder = autoencoder_with_regularization(
    #     None, None)
    # autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam',
                        loss='mse')
    # autoencoder.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    # Record losses and epochs during training
    num_epochs = 50
    epochs = []
    losses = []

    #  prepare  input data.
    # nColumns = data.shape[0]
    # xTrain = data.iloc[:round(nColumns*(4/5)), :]  # first 4/5 for training
    # xTest = data.iloc[round(nColumns*(4/5)):, :]  # final 1/5  for testing

    data2 = data
    noise = np.random.normal(0.0, scale=1.0, size=data2.shape)
    data2 = data2 + 0.05*noise
    xTrain = data2.sample(frac=0.8, random_state=42)
    print('xTrainSize:')
    print(xTrain.shape)
    xTest = data2.drop(xTrain.index)

    losses = []
    epochs = []

    # Callback to record loss and epochs during training

    class LossHistory(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epochs.append(epoch)
            losses.append(logs['loss'])

    autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                    validation_data=(xTest, xTest), callbacks=[LossHistory()])

    end_time = time.time()

    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs Epochs')
    plt.legend()
    plt.show()

    # Encode and decode some data
    predictedData = autoencoder.predict(data)
    mae = mean_absolute_error(data, predictedData)  # mean absolute error

    # Option 2: Time series
    # plt.figure(figsize=(20, 4))

    for i in range(4):
        # Display original
        ax = plt.subplot(2, 5, i + 1)
        plt.plot(xTest.iloc[:, i])
        # plt.ylim(0, 1)

        title = 'xmeas' + str(i)
        ax.set_title(title)

        # Display reconstruction
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.plot(predictedData[:, i])
        # plt.ylim(0, 1)
    plt.show()

    autoencoder.summary()
    print('Mean absolte error: ', mae)
    elapsed_time = end_time - start_time
    print("Elapsed time:", elapsed_time, "seconds")
    return losses


data = pd.read_csv("FaultLabeller/Data/Scenario1.csv")

data = data.drop(data.columns[[0]], axis=1)  # Remove extra columns
data = data.iloc[:, :52]

# data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column
print(data)
print(data.shape)

performAutoEncoding(data)
