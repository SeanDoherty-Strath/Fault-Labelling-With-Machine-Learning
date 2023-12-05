import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import keras
from keras import layers
import pandas as pd
import pyreadr
import math
from pathlib import Path
from sklearn.model_selection import KFold

n = 10000
numFolds = 5
kf = KFold(n_splits=numFolds, shuffle=True, random_state=42)
# Initialize an array to store the performance metrics for each fold
fold_metrics = []


def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


df = pd.read_csv(
    "TenesseeEastemen_FaultyTraining_Subsection.csv")
df = df.iloc[:, 3:]

rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
all_df = rdata_read["faulty_training"]
df = all_df.iloc[:n, 3:]


df = normalize_dataframe(df)


# Define the dimensions
input_output_dimension = 52
encoding_dimension = 12

# Define input layer
input_layer = keras.Input(shape=(input_output_dimension,))
# input_layer = layers.Dropout(0.2)(input_layer)

# Encoder
encoder = layers.Dense(25, activation='relu',)(input_layer)
encoder = layers.Dropout(0.2)(encoder)
encoder = layers.Dense(encoding_dimension, activation='relu')(encoder)


# Decoder
decoder = layers.Dense(25, activation='relu')(encoder)
decoder = layers.Dense(input_output_dimension, activation='sigmoid')(decoder)

# Create the autoencoder model
autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

# Compile the model
# autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.compile(optimizer='adam', loss='mse')

print(df)
# Replace headers with integer values for next step
# df.columns = range(len(df.columns))
# print(df)


for fold, (train_index, val_index) in enumerate(kf.split(df)):
    print(f"Training on fold {fold + 1}/{numFolds}")

    # xTrain = df.iloc[:math.floor(n*2/3), :]  # first 2 thirds for training
    # xTest = df.iloc[math.ceil(n*2/3):, :]  # final third for testing
    xTrain, xTest = df.iloc[train_index], df.iloc[val_index]

    # Train the autoencoder model on the current fold
    history = autoencoder.fit(
        xTrain, xTrain, epochs=50, validation_data=(xTest, xTest), verbose=2)

    # Evaluate the model on the validation set and store the metrics
    val_loss = autoencoder.evaluate(xTest, xTest, verbose=0)
    fold_metrics.append(val_loss)

average_loss = np.mean(fold_metrics)
print(f"Average validation loss across {numFolds} folds: {average_loss}")
