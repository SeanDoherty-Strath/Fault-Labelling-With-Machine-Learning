from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from keras import layers
import numpy as np
import keras
from keras import layers
import pandas as pd



# CLUSTERING FUNCTION
def performKMeans(df, k):

    # Normalize data
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    # Peform K-Means
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(data)

    # Get centoids and reutn labels
    labelArray = kmeans.labels_
    labels = labelArray.tolist()

    return labels


# Tune DBSCAN Parameters
def findKneePoint(df, k):

    # Normalize data
    scaler = StandardScaler()
    normalizedValues = scaler.fit_transform(df)

    # Find k nearest neighborus
    nearestNeighbours = NearestNeighbors(n_neighbors=k)
    nearestNeighbours.fit(normalizedValues)

    # calculate euclidian distance to k neighbors
    distances, _ = nearestNeighbours.kneighbors(normalizedValues)
    averageDistances = np.mean(distances, axis=1)

    #  sort into ascending order
    sortDistances = np.sort(averageDistances)

    # Calculate cdf
    cdf = np.cumsum(sortDistances)
    cdf /= cdf[-1]

    # find knee point - which corresponds to optimal value of EPS
    kneePointsIndex = np.argmax(cdf >= 0.9)
    eps = sortDistances[kneePointsIndex]

    return eps


def performDBSCAN(df, eps, minVal):
    # Normalize data
    scaler = StandardScaler()
    normalizedData = scaler.fit_transform(df)

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=minVal)
    dbscan.fit(normalizedData)
    labels = dbscan.labels_.tolist()

    # print('Number of labels from DBSCAN ', len(set(labels)))

    return labels



# FEATURE REDUCTION FUNCTIONS
def performPCA(df, n):

    # Normalize Data
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(df)

    # Peroform PCA
    pca = PCA(n_components=n)

    # Keep principal components
    principalComponents = pca.fit_transform(scaledData)

    # Create a new data frame
    columns = []
    for i in range(n):
        columns.append('PCA' + str(i))

    principalDF = pd.DataFrame(data=principalComponents, columns=columns)
    return principalDF


def createAutoencoder(df):

    # Standardize data
    scaler = StandardScaler()
    normalizedData = scaler.fit_transform(df)

    # Create new df with the normalized values
    trainingData = pd.DataFrame(
        normalizedData, columns=trainingData.columns)

    length = trainingData.shape[1]

    # Define the dimensions
    inputOutputDimensions = length
    hiddenLayerDimensions = round(length/2)  # Hidden layer is approx. half the previous
    encodingDimensions = 8


    inputLayer = keras.Input(shape=(inputOutputDimensions,))
    encoder = layers.Dense(hiddenLayerDimensions, activation='relu')(inputLayer)
    bottleneck = layers.Dense(encodingDimensions, activation='relu')(encoder)
    decoder = layers.Dense(hiddenLayerDimensions,activation='relu')(bottleneck)
    outputLayer = layers.Dense(inputOutputDimensions, activation='linear')(decoder)

    # AUTOENCODER
    autoencoder = keras.Model(inputs=inputLayer, outputs=outputLayer)

    # COMPILE MODEL
    autoencoder.compile(optimizer='adam', loss='mse')


    # note: Don't do cross fold valdiation, for time efficiency
    xTrain = trainingData.sample(frac=0.8, random_state=42)
    xTest = trainingData.drop(xTrain.index)

    autoencoder.fit(xTrain, xTrain, epochs=100, shuffle=True,validation_data=(xTest, xTest))

    bottleneck = keras.Model(inputs=autoencoder.input,outputs=autoencoder.get_layer('dense_1').output)

    # latentSpace = bottleneck.predict(normalizedData)
    # bottleneckDF = pd.DataFrame(data=latentSpace)
    # return bottleneckDF

    return bottleneck

