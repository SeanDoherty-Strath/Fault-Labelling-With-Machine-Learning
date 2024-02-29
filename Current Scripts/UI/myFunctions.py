import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import math
from sklearn.neighbors import NearestNeighbors
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("Data/UpdatedData.csv")
data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)  # Remove extra columns
# data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column


def changeText(text):
    return text


def updateGraph(value, data):
    fig = px.line(data, y=value)


sensors = []
for i in range(52):
    temp = 'Sensor' + str(i)
    sensors.append(temp)
    # print(sensors)

# print(sensors)


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

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k, n_init="auto")

    # Fit the model to the data
    kmeans.fit(df)

    # Get the cluster labels and centroids
    labelArray = kmeans.labels_
    labels = labelArray.tolist()

    return labels


def findBestParams(data, Range):
    # Define a range of epsilon values and min_samples values to search

    # eps_range = np.arange(1, 40, 5)
    eps_range = range(20, 50, 5)
    min_samples_range = range(3)

    best_score = -1
    best_eps = None
    best_min_samples = None

    # Perform grid search
    for eps_value in eps_range:
        for min_samples_value in min_samples_range:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            cluster_labels = dbscan.fit_predict(data)

            print('Eps: ', eps_value)
            print('Min: ', min_samples_value)
            numberOfLabels = len(list(set(cluster_labels)))

            print('Number of labels: ', numberOfLabels)
            if (numberOfLabels != 1):

                # Compute silhouette score
                score = silhouette_score(data, cluster_labels)
                print('Score: ', score)

            # Update best score and parameters if necessary
                minimumClusterSize = True
                myList = []
                # for i in range(min(cluster_labels), 1+max(cluster_labels)):
                #     myList = cluster_labels.tolist()
                #     if myList.count(i) < 20:
                #         minimumClusterSize = False
                if score > best_score and numberOfLabels >= 4 and numberOfLabels < 10 and minimumClusterSize:
                    # if score > best_score:
                    best_score = score
                    best_eps = eps_value
                    best_min_samples = min_samples_value

    print("Best silhouette score:", best_score)
    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)
    return best_eps, best_min_samples

    print("Best silhouette score:", best_score)
    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)


# def knee_point(X, k):
#     # Fit a k-nearest neighbor model
#     nn = NearestNeighbors(n_neighbors=k)
#     nn.fit(X)

#     # Compute distances to k-nearest neighbors
#     distances, _ = nn.kneighbors(X)
#     avg_distances = np.mean(distances, axis=1)

#     # Sort distances in ascending order
#     sorted_distances = np.sort(avg_distances)

#     # Calculate the first derivative
#     derivative = np.diff(sorted_distances)

#     # Find the knee point
#     knee_point_index = np.argmax(derivative)
#     knee_point_value = sorted_distances[knee_point_index]

#     return knee_point_value

def knee_point(X, k):
    # Fit a k-nearest neighbor model
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(X)

    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(normalized_values)

    # Compute distances to k-nearest neighbors
    distances, _ = nn.kneighbors(normalized_values)
    avg_distances = np.mean(distances, axis=1)

    # Sort distances in ascending order
    sorted_distances = np.sort(avg_distances)

    # Calculate the cumulative distribution function
    cdf = np.cumsum(sorted_distances)
    cdf /= cdf[-1]

    # Find the knee point
    knee_point_index = np.argmax(cdf >= 0.9)
    knee_point_value = sorted_distances[knee_point_index]

    return knee_point_value


def performDBSCAN(data, eps, minVal):

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    dbscan = DBSCAN(eps=eps, min_samples=minVal)
    dbscan.fit(normalized_values)
    labels = dbscan.labels_.tolist()
    print('Number of labels: ', len(set(labels)))
    return labels
#     labels = dbscan.labels_.tolist()
# def performDBSCAN(data, eps, minVal):
#     # Perform DBSCAN clustering

#     # short_df = data.iloc[::3]
#     print('Performing DBSCAN')
#     # Range = data.values.max() - data.values.min()

#     # best_eps, best_min_samples = findBestParams(data, Range)

#     dbscan = DBSCAN(eps=eps, min_samples=minVal)
#     dbscan.fit(data)
#     labels = dbscan.labels_.tolist()

#     # Add cluster labels to the dataset
#     print('Number of labels: ', len(set(labels)))
#     print('Min: ', min(set(labels)))
#     return labels


def performAutoEncoding(data):

    # Extract  values from df
    # data = pd.DataFrame(data_array)
    # rawValues = data.values()

    # Standardize data
    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    # Create new df with the normalized values
    data = pd.DataFrame(normalized_values, columns=data.columns)

    # return data
    length = data.shape[1]

    # Define the dimensions
    input_output_dimension = length
    hidden_layer_dimension = round(length/2)
    encoding_dimension = round(length/4)

    # INPUT LAYER
    input_layer = keras.Input(shape=(input_output_dimension,))
    # input_layer = layers.Dropout(0.2)(input_layer)

    # ENCODER
    encoder = layers.Dense(hidden_layer_dimension,
                           activation='relu',)(input_layer)
    # encoder = layers.Dropout(0.2)(encoder)
    encoder = layers.Dense(encoding_dimension, activation='relu')(encoder)

    # DECODER
    decoder = layers.Dense(hidden_layer_dimension, activation='relu')(encoder)
    # decoder = layers.Dropout(0.2)(decoder)
    decoder = layers.Dense(input_output_dimension,
                           activation='sigmoid')(decoder)

    # AUTOENCODER
    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

    # COMPILE MODEL
    autoencoder.compile(optimizer='adam', loss='mse')
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    # Record losses and epochs during training
    num_epochs = 50
    epochs = []
    losses = []

    #  prepare  input data.
    xTrain = data.iloc[:16000, :]  # first 4/5 for training
    xTest = data.iloc[16000:, :]  # final 1/5  for testing
    # This would be better by picking every 4th value or a random 4/5s

    losses = []
    epochs = []

    # Callback to record loss and epochs during training

    class LossHistory(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epochs.append(epoch)
            losses.append(logs['loss'])

    autoencoder.fit(xTrain, xTrain, epochs=num_epochs, shuffle=True,
                    validation_data=(xTest, xTest), callbacks=[LossHistory()])

    plt.plot(epochs, losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.show()

    # # Option 2: Time series
    # plt.figure(figsize=(20, 4))

    # for i in range(5):
    #     # Display original
    #     ax = plt.subplot(2, 5, i + 1)
    #     plt.plot(testDF.iloc[:960, i])
    #     plt.ylim(0, 1)

    #     title = 'xmeas' + str(i)
    #     ax.set_title(title)

    #     # Display reconstruction
    #     ax = plt.subplot(2, 5, i + 1 + 5)
    #     plt.plot(predictedData[:960, i])
    #     plt.ylim(0, 1)
    # plt.show()

    autoencoder.summary()
    # CONVERT TO LATENT SPACE

    bottleneck = keras.Model(inputs=autoencoder.input,
                             outputs=autoencoder.get_layer('dense_1').output)

    # Transform data into the latent space
    latentSpace = bottleneck.predict(data)

    bottleneck_df = pd.DataFrame(
        data=latentSpace)
    return bottleneck_df
