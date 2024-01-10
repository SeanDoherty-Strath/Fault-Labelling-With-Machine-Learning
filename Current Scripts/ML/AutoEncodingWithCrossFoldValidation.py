import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import keras
from keras import layers
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler


# READ IN DATA
csv = pd.read_csv("Data/UpdatedData.csv")
values = csv.values

# Standardize data
scaler = StandardScaler()
normalized_values = scaler.fit_transform(values)

# Create new df with the normalized values
df = pd.DataFrame(normalized_values, columns=csv.columns)

# remove first first columns, which are not sensor data
df = df.iloc[:, 4:]
print(df)
print(df.shape)
# 2000 x 52


# Define the neural network model
def create_model():

    # Define the dimensions
    input_output_dimension = 52
    hidden_layer_dimension = 25
    encoding_dimension = 12

    # INPUT LAYER
    input_layer = keras.Input(shape=(input_output_dimension,))
    # input_layer = layers.Dropout(0.2)(input_layer)

    # ENCODER
    encoder = layers.Dense(hidden_layer_dimension, activation='relu')(input_layer)
    encoder = layers.Dense(20, activation='relu')(encoder)

    # DECODER
    decoder = layers.Dense(encoding_dimension, activation='relu')(encoder)
    decoder = layers.Dense(input_output_dimension, activation='linear')(decoder)  # Use linear activation here

    # AUTOENCODER
    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

    # COMPILE MODEL
    autoencoder.compile(optimizer='adam', loss='mse')

    return autoencoder

# Specify the number of folds
k_folds = 5

# Initialize KFold object
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

# Lists to store training history
all_histories = []

# Perform K-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(df)):
    print(f"\nFold {fold + 1}/{k_folds}")

    # Split the data into training and validation sets for this fold
    xTrain, xVal = df.iloc[train_index], df.iloc[val_index]

    # Create and compile the model
    model = create_model()

    # Train the model on the current fold and store the training history
    history = model.fit(xTrain, xTrain, epochs=500, batch_size=10, validation_data=(xVal, xVal), verbose=0)
    
    # Evaluate the model on the validation set for this fold
    val_loss = model.evaluate(xVal, xVal)
    print(f"Validation Loss: {val_loss}")

    # Store the training history for this fold
    all_histories.append(history)

# Plot loss against epochs for each fold
plt.figure(figsize=(12, 6))
for fold, history in enumerate(all_histories):
    plt.plot(history.history['loss'], label=f'Fold {fold + 1} Training Loss')

plt.title('Training Loss Across Folds')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

predictedData = model.predict(df)

plt.figure(figsize=(20, 4))

for i in range(5):
    # Display original
    ax = plt.subplot(2, 5, i + 1)
    plt.plot(df.iloc[:960, i])
    plt.ylim(-3, 3)

    title = 'xmeas' + str(i)
    ax.set_title(title)

    # Display reconstruction
    ax = plt.subplot(2, 5, i + 1 + 5)
    plt.plot(predictedData[:960, i])
    plt.ylim(-3, 3)
plt.show()
