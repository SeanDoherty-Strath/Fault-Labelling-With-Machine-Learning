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
import time


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping


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
        normalizedData, columns=df.columns)

    length = trainingData.shape[1]

    # Define the dimensions
    inputOutputDimensions = length
    # Hidden layer is approx. half the previous
    hiddenLayerDimensions = round(length/2)
    encodingDimensions = 8

    inputLayer = keras.Input(shape=(inputOutputDimensions,))
    encoder = layers.Dense(hiddenLayerDimensions,
                           activation='relu')(inputLayer)
    bottleneck = layers.Dense(encodingDimensions, activation='relu')(encoder)
    decoder = layers.Dense(hiddenLayerDimensions,
                           activation='relu')(bottleneck)
    outputLayer = layers.Dense(
        inputOutputDimensions, activation='linear')(decoder)

    # AUTOENCODER
    autoencoder = keras.Model(inputs=inputLayer, outputs=outputLayer)

    # COMPILE MODEL
    autoencoder.compile(optimizer='adam', loss='mse')

    # note: Don't do cross fold valdiation, for time efficiency
    xTrain = trainingData.sample(frac=0.8, random_state=42)
    xTest = trainingData.drop(xTrain.index)

    early_stopping = EarlyStopping(
        monitor='mse', patience=10, verbose=1, restore_best_weights=True)

    autoencoder.fit(xTrain, xTrain, epochs=50, shuffle=True,
                    validation_data=(xTest, xTest), callbacks=[early_stopping])

    bottleneck = keras.Model(inputs=autoencoder.input,
                             outputs=autoencoder.get_layer('dense_1').output)

    # latentSpace = bottleneck.predict(normalizedData)
    # bottleneckDF = pd.DataFrame(data=latentSpace)
    # return bottleneckDF

    return bottleneck


def trainNeuralNetwork(trainingData):
    X = trainingData.iloc[:, :-1]
    y = trainingData.iloc[:, -1]
    y = np.array(y)

    print('got here 2')
    inputSize = X.shape[1]
    outputSize = len(set(y))

    # One-hot encode the target labels
    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(
            inputSize,)),  # 4 input features
        Dense(outputSize, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',  # Use categorical crossentropy for one-hot encoded labels
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    # Train the model
    # early_stopping = EarlyStopping(
    #     monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test))

    return model


def useNeuralNetwork(df, classifierNeuralNetwork):
    predictLabels = classifierNeuralNetwork.predict(df)

    # Round the highest value to 1 and all others to 0
    roundedLabels = np.zeros_like(predictLabels)
    roundedLabels[np.arange(len(predictLabels)),
                  predictLabels.argmax(axis=1)] = 1
    roundedLabels = np.argmax(roundedLabels, axis=1)
    for i in range(len(roundedLabels)):
        roundedLabels[i] += 1

    return roundedLabels


def testDBSCAN(predictedLabels):
    # DATA

    # Scenario 2
    #   - Normal operation 100 samples
    #   - Fault 1 for 20 samples
    #   - Normal operation 100 samples
    #   - Fault 2 for 20 samples
    # - Normal operation 100 samples
    #   - Fault 3 for 20 samples
    #   - Repeated three times
    combinations = []

    zeros = [0] * 100
    ones = [1] * 20
    twos = [2] * 20
    threes = [3] * 20

    combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
                        zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
    combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
                        zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
    combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
                        zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
    combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
                        zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
    combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
                        zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
    combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
                        zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

    zeros = [1] * 100
    ones = [0] * 20
    twos = [2] * 20
    threes = [3] * 20

    combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
                        zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
    combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
                        zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
    combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
                        zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
    combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
                        zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
    combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
                        zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
    combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
                        zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

    zeros = [2] * 100
    ones = [1] * 20
    twos = [0] * 20
    threes = [3] * 20

    combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
                        zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
    combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
                        zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
    combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
                        zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
    combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
                        zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
    combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
                        zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
    combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
                        zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

    zeros = [3] * 100
    ones = [1] * 20
    twos = [2] * 20
    threes = [0] * 20

    combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
                        zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
    combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
                        zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
    combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
                        zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
    combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
                        zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
    combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
                        zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
    combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
                        zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

    bestAccuracy = 0

    # testData = pd.read_csv("FaultLabeller/Data/Scenario1withLabels.csv")
    # correctLabels = testData.iloc[:, -1]

    for c in combinations:
        agreed_elements = 0
        for item1, item2 in zip(predictedLabels, c):
            if item1 == item2:
                agreed_elements += 1

        accuracy_percentage = (agreed_elements / len(c)) * 100

        if accuracy_percentage > bestAccuracy:
            bestAccuracy = accuracy_percentage

    print('ACCURACY: ', bestAccuracy)
