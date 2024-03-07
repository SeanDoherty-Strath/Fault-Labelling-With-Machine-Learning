import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
import time


# Normalise data


def performAutoencoding(data):
    # Define the dimensions
    # input_output_dimension = 52
    # hidden_layer_dimension = 25
    # encoding_dimension = 12

    scaler = StandardScaler()
    normalized_values = scaler.fit_transform(data)

    # Create new df with the normalized values
    data = pd.DataFrame(normalized_values, columns=data.columns)

    # INPUT LAYER
    input_layer = keras.Input(shape=(50,))
    # encoder = layers.Dense(hidden_layer_dimension, activation='relu',)(input_layer)
    latent_space = layers.Dense(12, activation='relu')(input_layer)
    # decoder = layers.Dense(hidden_layer_dimension, activation='relu')(encoder)
    output_layer = layers.Dense(50, activation='linear')(latent_space)

    # AUTOENCODER
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)

    # COMPILE MODEL
    autoencoder.compile(optimizer='adam', loss='mse')

    # K FOLD VALIDATION
    numFolds = 5
    kf = KFold(n_splits=numFolds, shuffle=True, random_state=42)
    # store performance metrics for each fold
    fold_metrics = []

    # Record losses and epochs during training
    num_epochs = 100
    epochs = []
    losses = []

    # Callback for graphing epochs

    class LossHistory(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epochs.append(epoch)
            losses.append(logs['loss'])

    # TRAINING
    for fold, (train_index, val_index) in enumerate(kf.split(data)):
        print(f"Training on fold {fold + 1}/{numFolds}")

        xTrain, xTest = data.iloc[train_index], data.iloc[val_index]

        # Train the autoencoder model on the current fold
        autoencoder.fit(xTrain, xTrain, epochs=num_epochs, batch_size=20, validation_data=(
            xTest, xTest), verbose=2, callbacks=[LossHistory()])

        # Evaluate the model on the validation set and store the metrics
        val_loss = autoencoder.evaluate(xTest, xTest, verbose=0)
        fold_metrics.append(val_loss)

    average_loss = np.mean(fold_metrics)
    print(f"Average validation loss across {numFolds} folds: {average_loss}")

    plt.plot(epochs[:num_epochs], losses[:num_epochs], label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show()

    # # TEST THE DATA ON THE TEST SET
    # testDF = pd.read_csv(
    #     "Data/UpdatedTestData.csv")
    # testDF = normalize_dataframe(testDF)
    # testDF = testDF.iloc[:, 4:]

    predictedData = autoencoder.predict(data)
    mae = mean_absolute_error(data, predictedData)  # mean absolute error
    print('Mean absolte error: ', mae)

    # Option 2: Time series
    plt.figure(figsize=(20, 4))

    for i in range(5):
        # Display original
        ax = plt.subplot(2, 5, i + 1)
        plt.plot(data.iloc[:, i])

        title = 'xmeas' + str(i)
        ax.set_title(title)

        # Display reconstruction
        ax = plt.subplot(2, 5, i + 1 + 5)
        plt.plot(predictedData[:, i])
    plt.show()


data = pd.read_csv("FaultLabeller/Data/OperatingScenario1.csv")
data = data.drop(data.columns[[0]], axis=1)  # Remove extra columns
# data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column

data = data.iloc[:, :50]
print(data.shape)
print(data)

start_time = time.time()
performAutoencoding(data)
end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time:", elapsed_time, "seconds")
