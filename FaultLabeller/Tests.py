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
import tensorflow as tf
import tensorflow_addons as tfa
import time


def performAutoEncoding(data):
    # Standardize data
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    data = pd.DataFrame(normalized_values, columns=data.columns)

    input_layer = keras.Input(shape=(52,))
    encoder = layers.Dense(26, activation='relu')(input_layer)
    hidden_layer = layers.Dense(16, activation='relu')(encoder)
    decoder = layers.Dense(26, activation='relu')(hidden_layer)
    output_layer = layers.Dense(52, activation='linear')(decoder)

    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.compile(optimizer='adam', loss=tf.keras.losses.Huber())
    # Record losses and epochs during training
    num_epochs = 200

    data2 = data
    xTrain = data2.sample(frac=0.8, random_state=42)
    xTest = data2.drop(xTrain.index)

    losses = []
    epochs = []
    times = []

    # Callback to record loss and epochs during training

    start_time = time.time()

    class LossHistory(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):

            epochs.append(epoch)
            losses.append(logs['loss'])
            elapsed_time = time.time() - start_time
            times.append(elapsed_time)

    autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                    validation_data=(xTest, xTest), callbacks=[LossHistory()])

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Mean Squared Error', color=color)
    ax1.plot(epochs, losses, color=color, label='Training Loss')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    # we already handled the x-label with ax1
    ax2.set_ylabel('Time (s)', color=color)
    ax2.plot(epochs, times, color=color, label='Training Time')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend()
    plt.show()

    # Encode and decode some data
    predictedData = autoencoder.predict(data)
    mae = mean_absolute_error(data, predictedData)  # mean absolute error

    autoencoder.summary()
    print('Mean absolte error: ', mae)
    return losses


data = pd.read_csv("FaultLabeller/Data/Scenario1.csv")

data = data.drop(data.columns[[0]], axis=1)  # Remove extra columns
data = data.iloc[:, :52]

# data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column
print(data)
print(data.shape)

performAutoEncoding(data)
