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


def performAutoEncoding(data):
    start_time = time.time()
    # Standardize data
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    # Create new df with the normalized values
    data = pd.DataFrame(normalized_values, columns=data.columns)

    # return data
    # length = data.shape[1]

    # # Define the dimensions
    # input_output_dimension = 52
    # # hidden_layer_dimension =
    # encoding_dimension = 13

    # INPUT LAYER
    input_layer = keras.Input(shape=(50,))
    encoder = layers.Dense(25, activation='relu',)(input_layer)
    hidden_layer = layers.Dense(12, activation='relu')(encoder)
    decoder = layers.Dense(25, activation='relu')(hidden_layer)
    output_layer = layers.Dense(50, activation='linear')(decoder)

    # AUTOENCODER
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

    # COMPILE MODEL
    # autoencoder.compile(optimizer='adam', loss='mse')

    autoencoder.compile(optimizer='adam', loss='mse')

    # Record losses and epochs during training
    num_epochs = 100
    epochs = []
    losses = []

    #  prepare  input data.
    # nColumns = data.shape[0]
    # xTrain = data.iloc[:round(nColumns*(4/5)), :]  # first 4/5 for training
    # xTest = data.iloc[round(nColumns*(4/5)):, :]  # final 1/5  for testing
    data2 = data.iloc[:, :]
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
    plt.figure(figsize=(20, 4))

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


data = pd.read_csv("FaultLabeller/Data/OperatingScenario1.csv")
data = data.drop(data.columns[[0]], axis=1)  # Remove extra columns

# data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column


data = data.iloc[:, :50]
print(data.shape)
print(data)

performAutoEncoding(data)
