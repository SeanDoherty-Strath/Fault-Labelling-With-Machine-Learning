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
import tensorflow as tf


def performPCA(df, n):
    print('Perform PCA')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(scaled_data)

    columns = []
    for i in range(n):

        columns.append('PCA' + str(i))
    principal_df = pd.DataFrame(
        data=principal_components, columns=columns)
    return principal_df


def performKMeans(df, k):
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=k, n_init="auto")

    # Fit the model to the data
    kmeans.fit(data)

    # Get the cluster labels and centroids
    labelArray = kmeans.labels_
    labels = labelArray.tolist()

    return labels


def findKneePoint(df, k):

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(df)

    nearest_neigbours = NearestNeighbors(n_neighbors=k)
    nearest_neigbours.fit(normalized_values)

    # Compute euclidian distance to k neighbors
    distances, _ = nearest_neigbours.kneighbors(normalized_values)
    avg_distances = np.mean(distances, axis=1)

    # then sort into ascending order
    sortDistances = np.sort(avg_distances)

    # Calculate cdf
    cdf = np.cumsum(sortDistances)
    cdf /= cdf[-1]

    # finally, find knee point - which is the optimal value of EPS
    knee_point_index = np.argmax(cdf >= 0.9)
    eps = sortDistances[knee_point_index]

    return eps


def performDBSCAN(data, eps, minVal):

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    dbscan = DBSCAN(eps=eps, min_samples=minVal)
    dbscan.fit(normalized_values)
    labels = dbscan.labels_.tolist()
    print('Number of labels: ', len(set(labels)))
    return labels


def createAutoencoder(trainingData):

    # Standardize data
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(trainingData)

    # Create new df with the normalized values
    trainingData = pd.DataFrame(
        normalized_values, columns=trainingData.columns)

    length = trainingData.shape[1]

    # Define the dimensions
    input_output_dimension = length
    hidden_layer_dimension = round(length/2)
    encoding_dimension = 8

    input_layer = keras.Input(shape=(input_output_dimension,))
    encoder = layers.Dense(hidden_layer_dimension, activation='relu',
                           activity_regularizer=tf.keras.regularizers.l1(0.05))(input_layer)
    hidden_layer = layers.Dense(encoding_dimension, activation='relu',
                                activity_regularizer=tf.keras.regularizers.l1(0.05))(encoder)
    decoder = layers.Dense(hidden_layer_dimension,
                           activation='relu')(hidden_layer)
    output_layer = layers.Dense(
        input_output_dimension, activation='linear')(decoder)

    # AUTOENCODER
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

    # COMPILE MODEL
    autoencoder.compile(optimizer='adam', loss='mse')

#     # Record losses and epochs during training
    num_epochs = 50
#     epochs = []
#     losses = []
    #   class LossHistory(keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs=None):
#             epochs.append(epoch)
#             losses.append(logs['loss'])

    xTrain = trainingData.sample(frac=0.8, random_state=42)
    xTest = trainingData.drop(xTrain.index)

    autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                    validation_data=(xTest, xTest))

#     # plt.plot(epochs, losses, label='Training Loss')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('Loss')
#     # plt.title('Loss vs. Epochs')
#     # plt.legend()
#     # plt.show()

#     # create a model to convert to latent space

    bottleneck = keras.Model(inputs=autoencoder.input,
                             outputs=autoencoder.get_layer('dense_1').output)
    return bottleneck

    # Transform data into the latent space
    # latentSpace = bottleneck.predict(data)

#     bottleneck_df = pd.DataFrame(
#         data=latentSpace)
#     return bottleneck_df


# def performAutoEncoding(data):

#     # Standardize data
#     scaler = StandardScaler()
#     normalized_values = scaler.fit_transform(data)

#     # Create new df with the normalized values
#     data = pd.DataFrame(normalized_values, columns=data.columns)

#     # # return data
#     length = data.shape[1]

#     # Define the dimensions
#     input_output_dimension = length
#     hidden_layer_dimension = round(length/2)
#     encoding_dimension = 8

#     input_layer = keras.Input(shape=(input_output_dimension,))
#     # input_layer = layers.Dropout(0.01)(input_layer)
#     encoder = layers.Dense(
#         hidden_layer_dimension, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.05))(input_layer)
#     # encoder = layers.Dropout(0.01)(encoder)
#     hidden_layer = layers.Dense(
#         encoding_dimension, activation='relu', activity_regularizer=tf.keras.regularizers.l1(0.05))(encoder)
#     decoder = layers.Dense(hidden_layer_dimension,
#                            activation='relu')(hidden_layer)
#     output_layer = layers.Dense(
#         input_output_dimension, activation='linear')(decoder)

#     # AUTOENCODER
#     autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

#     # COMPILE MODEL
#     autoencoder.compile(optimizer='adam', loss='mse')

#     # Record losses and epochs during training
#     num_epochs = 50
#     epochs = []
#     losses = []

#     #  prepare  input data.
#     # xTrain = data.iloc[:16000, :]  # first 4/5 for training
#     # xTest = data.iloc[16000:, :]  # final 1/5  for testing

#     xTrain = data.sample(frac=0.8, random_state=42)
#     xTest = data.drop(xTrain.index)

#     losses = []
#     epochs = []

#     # Callback to record loss and epochs during training

#     class LossHistory(keras.callbacks.Callback):
#         def on_epoch_end(self, epoch, logs=None):
#             epochs.append(epoch)
#             losses.append(logs['loss'])

#     autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
#                     validation_data=(xTest, xTest), callbacks=[LossHistory()])

#     # plt.plot(epochs, losses, label='Training Loss')
#     # plt.xlabel('Epochs')
#     # plt.ylabel('Loss')
#     # plt.title('Loss vs. Epochs')
#     # plt.legend()
#     # plt.show()

#     autoencoder.summary()
#     # CONVERT TO LATENT SPACE

#     bottleneck = keras.Model(inputs=autoencoder.input,
#                              outputs=autoencoder.get_layer('dense_1').output)

#     # Transform data into the latent space
#     latentSpace = bottleneck.predict(data)

#     bottleneck_df = pd.DataFrame(
#         data=latentSpace)
#     return bottleneck_df
