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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
from tensorflow.keras.callbacks import EarlyStopping


# CLUSTERING FUNCTION
def performKMeans(df, k):

    # Normalize data
    scaler = StandardScaler()
    data = scaler.fit_transform(df)

    # Peform K-Means
    kmeans = KMeans(n_clusters=k, n_init="auto")
    kmeans.fit(data)

    # Return Labels
    return kmeans.labels_.tolist()


# Tune DBSCAN Parameters
def findKneePoint(df, k):

    # Normalize data
    scaler = StandardScaler()
    normalizedValues = scaler.fit_transform(df)

    # Create K Plot using neurest neighbours
    NN = NearestNeighbors(n_neighbors=k)
    NN.fit(normalizedValues)
    eucDistancec, _ = NN.eucDistancec(normalizedValues)  # use euc distance
    avgEucDiistance = np.mean(eucDistancec, axis=1)
    sortedEucDistance = np.sort(avgEucDiistance)  # Go into ascendng order

    # Calculate cdf
    cdf = np.cumsum(sortedEucDistance)
    cdf /= cdf[-1]

    # find knee point (corresponds to optimal value of epsiilon)
    kneedPointIndex = np.argmax(cdf >= 0.9)
    eps = sortedEucDistance[kneedPointIndex]

    return eps


def performDBSCAN(df, eps, minVal):
    # Normalize data
    scaler = StandardScaler()
    normalizedData = scaler.fit_transform(df)

    # Perform DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=minVal)
    dbscan.fit(normalizedData)

    # print('Number of labels from DBSCAN ', len(set(labels)))

    return dbscan.labels_.tolist()


# FEATURE REDUCTION FUNCTIONS
def performPCA(df, n):

    # Normalize Data
    scaler = StandardScaler()
    scaledData = scaler.fit_transform(df)

    # Perform PCA PCA
    pca = PCA(n_components=n)
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
    trainingData = pd.DataFrame(normalizedData, columns=df.columns)

    # Set layer dimensions
    nInputs = trainingData.shape[1]
    inputOutputDimensions = nInputs
    # Hidden layer is approx. half the previous
    hiddenLayerDimensions = round(nInputs/2)
    encodingDimensions = 8

    # Build Autoencoder
    inputLayer = keras.Input(shape=(inputOutputDimensions,))
    encoder = layers.Dense(hiddenLayerDimensions,
                           activation='relu')(inputLayer)
    bottleneck = layers.Dense(encodingDimensions, activation='relu')(encoder)
    decoder = layers.Dense(hiddenLayerDimensions,
                           activation='relu')(bottleneck)
    outputLayer = layers.Dense(
        inputOutputDimensions, activation='linear')(decoder)

    # Create & Traun Autoencoder
    autoencoder = keras.Model(inputs=inputLayer, outputs=outputLayer)
    autoencoder.compile(optimizer='adam', loss='mse')
    xTrain = trainingData.sample(frac=0.8, random_state=42)
    xTest = trainingData.drop(xTrain.index)
    autoencoder.fit(xTrain, xTrain, epochs=50, shuffle=True,
                    validation_data=(xTest, xTest))

    # Pass data to Bottleneck
    bottleneck = keras.Model(inputs=autoencoder.input,
                             outputs=autoencoder.get_layer('dense_1').output)
    return bottleneck


def trainNeuralNetwork(trainingData):
    # Inputs
    X = trainingData.iloc[:, :-1]
    inputSize = X.shape[1]

    # Outputs
    y = np.array(trainingData.iloc[:, -1])
    outputSize = len(set(y))
    encoder = OneHotEncoder(sparse=False)  # One-Hot Encoder
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split randomly into train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Dense(64, activation='elu', input_shape=(inputSize,)),
        Dense(outputSize, activation='softmax')
    ])

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # Train the model
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
    model.fit(X_train, y_train, epochs=50, batch_size=32,
              validation_data=(X_test, y_test), callbacks=[early_stopping])

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


def testAccuracy(predictedLabels):
    # Read in the test scenario
    testData = pd.read_csv("FaultLabeller/Data/Scenario1withLabels.csv")
    correctLabels = testData.iloc[:, -1]

    score = 0

    for i in range(len(correctLabels)):
        if correctLabels[i] == predictedLabels[i]:
            score += 1
    print('ACCURACY: ', score / len(correctLabels))
