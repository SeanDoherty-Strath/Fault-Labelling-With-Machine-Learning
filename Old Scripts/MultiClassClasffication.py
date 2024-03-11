import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import keras
# Load Iris dataset
# iris = load_iris()
# X = iris.data
# y = iris.target

X = pd.read_csv("FaultLabeller/Data/OperatingScenario6.csv")
X = X.drop(X.columns[[0]], axis=1)  # Remove extra columns

y = [0]*480 + [1]*480 + [2]*480 + [3]*480
y = y*10
y = np.array(y)


# One-hot encode the target labels
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y.reshape(-1, 1))
print(y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(52,)),  # 4 input features
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),  # Input layer with 52 neurons
    Dense(4, activation='softmax')  # Output layer with 3 units for 3 classes
])

# Compile the model
model.compile(optimizer='adam',
              # Use categorical crossentropy for one-hot encoded labels
              loss='categorical_crossentropy',
              metrics=['accuracy'])

epochs = []
losses = []


class LossHistory(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        epochs.append(epoch)
        losses.append(logs['loss'])


# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32,
          validation_data=(X_test, y_test), callbacks=[LossHistory()])


plt.plot(epochs, losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs. Epochs')
plt.legend()
plt.show()


# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)

print('Test accuracy:', test_acc)

predicted = model.predict(X)
# Round the highest value to 1 and all others to 0
rounded_predictions = np.zeros_like(predicted)
rounded_predictions[np.arange(len(predicted)), predicted.argmax(axis=1)] = 1

print(rounded_predictions)
plt.plot(rounded_predictions)
plt.show()
