import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import plotly.express as px

# Generate some example data
np.random.seed(42)
X = np.random.rand(1000, 10)  # 1000 samples, each with 10 features

# Split the data into training and testing sets
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

# Normalize the data using Min-Max scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the autoencoder model using scikit-learn's MLPRegressor
# The hidden layer size determines the size of the encoded representation
autoencoder = MLPRegressor(hidden_layer_sizes=(
    5,), max_iter=1000, random_state=42)

# Train the autoencoder on the training data
autoencoder.fit(X_train_scaled, X_train_scaled)

# Encode and decode the test data
X_test_encoded = autoencoder.transform(X_test_scaled)
X_test_decoded = autoencoder.inverse_transform(X_test_encoded)

# Plot some examples of original and decoded data
n_examples = 5
fig, axes = px.subplots(nrows=2, ncols=n_examples, figsize=(10, 4))

for i in range(n_examples):
    axes[0, i].imshow(X_test_scaled[i].reshape(1, -1), cmap='gray')
    axes[1, i].imshow(X_test_decoded[i].reshape(1, -1), cmap='gray')

axes[0, 0].set_ylabel('Original')
axes[1, 0].set_ylabel('Decoded')

px.tight_layout()
px.show()
