import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import layers
import pandas as pd
from pathlib import Path
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler

# Normalise data


def normalize_dataframe(df):
    normalized_df = pd.DataFrame()
    for col in df.columns:  # Ignore the first four columns
        if df.columns.get_loc(col) < 4:
            normalized_df[col] = df[col]
        else:  # Noramlize everything else between 0 and 1
            normalized_col = (df[col] - df[col].min()) / \
                (df[col].max() - df[col].min())
            normalized_df[col] = normalized_col

    return normalized_df


# Read in data
df_raw = pd.read_csv(
    "Data/UpdatedData.csv")

# Extract  values from df
rawValues = df_raw.values

# Standardize data
scaler = StandardScaler()
normalized_values = scaler.fit_transform(rawValues)

# Create new df with the normalized values
df = pd.DataFrame(normalized_values, columns=df_raw.columns)

# remove first first columns
df = df.iloc[:, 4:]
print(df)
print(df.shape)
# 2000 x 52

# Define the dimensions
input_output_dimension = 52
hidden_layer_dimension = 25
encoding_dimension = 12

# INPUT LAYER
input_layer = keras.Input(shape=(input_output_dimension,))
# input_layer = layers.Dropout(0.2)(input_layer)

# ENCODER
encoder = layers.Dense(hidden_layer_dimension, activation='relu',)(input_layer)
# encoder = layers.Dropout(0.2)(encoder)
encoder = layers.Dense(20, activation='relu')(encoder)

# DECODER
decoder = layers.Dense(hidden_layer_dimension, activation='relu')(encoder)
# decoder = layers.Dropout(0.2)(decoder)
decoder = layers.Dense(input_output_dimension, activation='sigmoid')(decoder)

# AUTOENCODER
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# COMPILE MODEL
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# autoencoder.compile(optimizer='adam', loss='mse')

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
for fold, (train_index, val_index) in enumerate(kf.split(df)):
    print(f"Training on fold {fold + 1}/{numFolds}")

    xTrain, xTest = df.iloc[train_index], df.iloc[val_index]

    # Train the autoencoder model on the current fold
    history = autoencoder.fit(
        xTrain, xTrain, epochs=num_epochs, batch_size=20, validation_data=(xTest, xTest), verbose=2, callbacks=[LossHistory()])

    # Evaluate the model on the validation set and store the metrics
    val_loss = autoencoder.evaluate(xTest, xTest, verbose=0)
    fold_metrics.append(val_loss)

average_loss = np.mean(fold_metrics)
print(f"Average validation loss across {numFolds} folds: {average_loss}")

plt.plot(epochs[:num_epochs], losses[:num_epochs], label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.show()


# TEST THE DATA ON THE TEST SET
testDF = pd.read_csv(
    "Data/UpdatedTestData.csv")
testDF = normalize_dataframe(testDF)
testDF = testDF.iloc[:, 4:]

predictedData = autoencoder.predict(testDF)
mae = mean_absolute_error(testDF, predictedData)  # mean absolute error
print('Mean absolte error: ', mae)

# Option 2: Time series
plt.figure(figsize=(20, 4))

for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(testDF.iloc[:960, i])
    plt.ylim(0, 1)

    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(predictedData[:960, i])
    plt.ylim(0, 1)
plt.show()
