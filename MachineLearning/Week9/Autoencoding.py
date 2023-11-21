
# The purpose of this code is to practise implmeneinng an auto encoder.
# I start with four inputs.  I have three hidden layers with 3, 2, and then 3 neurons
# Their are then four outputs. These are a recreation of the four inputs.
# We can verify their closesness by x' - x

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Take test data from iris
# This has four inputs (to do with petal widths etc)
iris = load_iris()
X = iris.data
X_prime = iris.data  # I've kept these seperate so it's clearer whats going on

# Split into training and testing
X_train, X_test, x_prime_train, x_prime_test = train_test_split(
    X, X_prime, test_size=0.2, random_state=42)

# Design small neural network
# hidden_layer_sizes parameter determines # of hidden layers and neurons in each
model = MLPClassifier(hidden_layer_sizes=(3, 2, 3),
                      max_iter=1000, random_state=42)

# Train model!
model.fit(X, X_prime)

# Make predictions test set
x_prime_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(x_prime_test, x_prime_pred)
print(f"Accuracy: {accuracy}")

# SKIP THIS FOR NOW
# Get the final biases and weights...
# final_weights = model.coefs_
# final_biases = model.intercepts_

# Print them out!
# print("Weights:")
# for j, w in enumerate(final_weights):
#    print(f"Layer {j}:\n{w}")

# print("Biases:")
# for j, b in enumerate(final_biases):
#    print(f"Layer {j}:\n{b}")
