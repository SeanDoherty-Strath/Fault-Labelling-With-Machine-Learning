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
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import math
from sklearn.metrics import accuracy_score


def performAutoEncoding(data, n):
    # Standardize data
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)
    data = pd.DataFrame(normalized_values, columns=data.columns)

    trainingData = data
    xTrain = trainingData.sample(frac=0.8, random_state=42)
    xTest = trainingData.drop(xTrain.index)

    num_epochs = 200

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

    if n == 0:
        input_layer = keras.Input(shape=(52,))
        encoder = layers.Dense(26, activation='relu')(input_layer)
        hidden_layer = layers.Dense(4, activation='relu')(encoder)
        decoder = layers.Dense(26, activation='relu')(hidden_layer)
        output_layer = layers.Dense(52, activation='linear')(decoder)
        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                        validation_data=(xTest, xTest), callbacks=[LossHistory()])
        predictedData = autoencoder.predict(data)
        mae = mean_absolute_error(data, predictedData)  # mean absolute error
        return mae, times[-1]/50
    if n == 1:
        input_layer = keras.Input(shape=(52,))
        encoder = layers.Dense(26, activation='relu')(input_layer)
        hidden_layer = layers.Dense(8, activation='relu')(encoder)
        decoder = layers.Dense(26, activation='relu')(hidden_layer)
        output_layer = layers.Dense(52, activation='linear')(decoder)
        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                        validation_data=(xTest, xTest), callbacks=[LossHistory()])
        predictedData = autoencoder.predict(data)
        mae = mean_absolute_error(data, predictedData)  # mean absolute error

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Error', color=color)
        ax1.plot(epochs, losses, color=color, label='Training Loss')
        ax1.tick_params(axis='y', labelcolor=color)
        # ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        # color = 'tab:blue'
        # # we already handled the x-label with ax1
        # ax2.set_ylabel('Time (s)', color=color)
        # ax2.plot(epochs, times, color=color, label='Training Time')    # ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        fig.legend()
        plt.show()

        return mae, times[-1]/50
    if n == 3:
        input_layer = keras.Input(shape=(52,))
        encoder = layers.Dense(26, activation='relu')(input_layer)
        hidden_layer = layers.Dense(12, activation='relu')(encoder)
        decoder = layers.Dense(26, activation='relu')(hidden_layer)
        output_layer = layers.Dense(52, activation='linear')(decoder)
        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                        validation_data=(xTest, xTest), callbacks=[LossHistory()])
        predictedData = autoencoder.predict(data)
        mae = mean_absolute_error(data, predictedData)  # mean absolute error
        return mae, times[-1]/50
    if n == 4:
        input_layer = keras.Input(shape=(52,))
        encoder = layers.Dense(30, activation='relu')(input_layer)
        hidden_layer = layers.Dense(16, activation='relu')(encoder)
        decoder = layers.Dense(30, activation='relu')(hidden_layer)
        output_layer = layers.Dense(52, activation='linear')(decoder)
        autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
        autoencoder.compile(optimizer='adam', loss='mse')
        autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                        validation_data=(xTest, xTest), callbacks=[LossHistory()])
        predictedData = autoencoder.predict(data)
        mae = mean_absolute_error(data, predictedData)  # mean absolute error
        return mae, times[-1]/50

    else:
        return 0, 0


data = pd.read_csv("FaultLabeller/Data/Scenario5.csv")

data = data.drop(data.columns[[0]], axis=1)  # Remove extra columns

print(data)
print(data.shape)


labels = np.random.rand(3, 3)  # Generate random data for demonstration

# loss = ['mse', 'mae', 'huber']
# optimization = ['adam', 'rmsprop', 'adagrad']
# df = pd.DataFrame(labels, columns=optimization)
# df.iloc[:, :] = 0

# timeArray = []
# accuracyArray = []

tempAccuracy, tempTime = performAutoEncoding(data, 1)
#         accuracy += tempAccuracy


# for n in range(5):
#     accuracy = 0.0
#     times = 0.0
#     for i in range(5):
#         tempAccuracy, tempTime = performAutoEncoding(data, n)
#         accuracy += tempAccuracy
#         times += tempTime
#     accuracy /= 5
#     times /= 5
#     accuracyArray.append(accuracy)
# #     timeArray.append(times)

# print(accuracyArray)
# print(timeArray)


# for i in range(len(optimization)):
#     for j in range(len(loss)):
#         accuracy = 0.0
#         for k in range(5):
#             temp = performAutoEncoding(data, optimization[i], loss[j])
#             accuracy += temp

#         accuracy = accuracy / 5
#         df.iloc[j, i] = accuracy

# print(df)

# labels = np.random.rand(6, 6)  # Generate random data for demonstration

# # Create colum names
# # activationFunctions = ['relu', 'softmax']
# activationFunctions = ['relu', 'sigmoid', 'tanh', 'elu', 'linear', 'softmax']

# # Create the DataFrame
# df = pd.DataFrame(labels, columns=activationFunctions)

# # activationFunctions = ['relu', 'sigmoid', 'tanh', 'elu', 'linear', 'softmax']


# for i in range(len(activationFunctions)):
#     for j in range(len(activationFunctions)):
#         accuracy = 0.0
#         for k in range(5):
#             temp = performAutoEncoding(data,
#                                        activationFunctions[i], activationFunctions[j])
#             accuracy += temp

#         accuracy = accuracy / 5
#         df.iloc[i, j] = accuracy
