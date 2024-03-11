import numpy as np
import tensorflow as tf
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import keras
import pandas as pd

import matplotlib.pyplot as plt

# data = FaultLabeller/Data/OperatingScenario5.csv
X = pd.read_csv("FaultLabeller/Data/OperatingScenario5.csv")
X = X.drop(X.columns[[0]], axis=1)  # Remove extra columns
X2 = X.iloc[4800:9600, :]
X = X.iloc[:4800, :]
zeros = [0]*480
ones = [1]*480
y = zeros+ones+zeros+ones+zeros+ones+zeros+ones+zeros+ones

print(X)
print(X.shape)
# Convert the list to a NumPy ndarray
y = np.array(y)

# Load the Breast Cancer Wisconsin Dataset
# cancer = load_breast_cancer()
# X = cancer.data
# y = cancer.target
# print(type(y))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential()
model.add(Dense(64, input_dim=52, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])

losses = []
epochs = []


class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epochs.append(epoch)
        losses.append(logs['loss'])


# Train the model
model.fit(X_train, y_train, epochs=150, batch_size=32,
          validation_split=0.2, callbacks=[LossHistory()])

plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()


# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')


predictData = model.predict(X2)
print(predictData)
plt.plot(predictData)
plt.show()
