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

from sklearn.metrics import accuracy_score


def peformNN(trainingData):
    # Standardize data
    # scaler = StandardScaler()
    # normalized_values = scaler.fit_transform(data)

    # data = pd.DataFrame(normalized_values, columns=data.columns)

    if 'Unnamed: 0' in trainingData.columns:
        trainingData.drop(columns=['Unnamed: 0'], inplace=True)

    if 'Time' in trainingData.columns:
        trainingData.drop(columns=['Time'], inplace=True)

    X = trainingData.iloc[:, :-1]
    y = trainingData.iloc[:, -1]
    y = np.array(y)
    inputSize = X.shape[1]
    outputSize = len(set(y))

    encoder = OneHotEncoder(sparse=False)
    y = encoder.fit_transform(y.reshape(-1, 1))

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Define the model
    model = Sequential([
        Dense(64, activation='elu', input_shape=(
            inputSize,)),  # 4 input features
        Dense(64, activation='elu'),  # 4 input features
        Dense(64, activation='elu'),  # 4 input features
        Dense(64, activation='elu'),  # 4 input features
        Dense(64, activation='elu'),  # 4 input features
        # Dense(64, activation='relu'),  # 4 input features
        Dense(outputSize, activation='softmax')
    ])

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

    loss_history = LossHistory()
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)

    # Compile the model
    model.compile(optimizer='adam',  # Use categorical crossentropy for one-hot encoded labels
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # Train the model
    history = model.fit(X_train, y_train, epochs=200, batch_size=32,
                        validation_data=(X_test, y_test), callbacks=[loss_history, early_stopping])

    fig, ax1 = plt.subplots()

    color = 'tab:red'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Catergorical Crossentropoy', color=color)
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
    # plt.show()

    # THEN TEST PERFORMANCE
    testData = pd.read_csv("FaultLabeller/Data/Scenario5withLabels.csv")
    if 'Unnamed: 0' in testData.columns:
        testData.drop(columns=['Unnamed: 0'], inplace=True)

    if 'Time' in testData.columns:
        testData.drop(columns=['Time'], inplace=True)

    actualLabels = testData.iloc[:, -1]
    testData = testData.iloc[:, :-1]

    predictedLabels = model.predict(testData)

    roundedLabels = np.zeros_like(predictedLabels)
    roundedLabels[np.arange(len(predictedLabels)),
                  predictedLabels.argmax(axis=1)] = 1
    predictedLabels = np.argmax(roundedLabels, axis=1)

    for i in range(len(predictedLabels)):
        predictedLabels[i] += 1

    actualLabels = np.array(actualLabels)

    plt.figure()
    plt.plot(predictedLabels)
    # plt.show()

    conf_matrix = confusion_matrix(actualLabels, predictedLabels)

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(actualLabels, predictedLabels))

    # Visualize confusion matrix
    labels = ['No Fault', 'Fault 1', 'Fault 2', 'Fault 3']
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()

    accuracy = accuracy_score(actualLabels, predictedLabels)
    print('Accuracy = ', accuracy)
    return accuracy, len(epochs),  times[-1]


trainingData = pd.read_csv("FaultLabeller/Data/Scenario1withLabels.csv")

trainingData = trainingData.drop(
    trainingData.columns[[0]], axis=1)  # Remove extra columns


# data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column
accuracy = [0, 0, 0, 0, 0]
epochs = [0, 0, 0, 0, 0]
times = [0, 0, 0, 0, 0]

for i in range(5):
    accuracy[i], epochs[i], times[i] = peformNN(trainingData)

mean = 0
avgEpochs = 0
avgTimes = 0
for i in range(5):
    mean += accuracy[i]
    avgEpochs += epochs[i]
    avgTimes += times[i]


mean = mean / 5
avgEpochs = avgEpochs / 5
avgTimes = avgTimes / 5
print('Avg accuracy: ', mean)
print('Avg Epochs: ', avgEpochs)
print('Avg Time: ', avgTimes)
