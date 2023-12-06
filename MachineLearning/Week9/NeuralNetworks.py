
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris

# Take test data from iris (see SK Learn)
iris = load_iris()
X = iris.data
y = iris.target

print('Inputs', X)
print('Outputs', y)

# Split into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Design small neural network
# hidden_layer_sizes parameter determines # of hidden layers and neurons in each
model = MLPClassifier(hidden_layer_sizes=(5,), max_iter=1000, random_state=42)

# Train model!
model.fit(X_train, y_train)

# Make predictions test set
y_pred = model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Get the final biases and weights...
final_weights = model.coefs_
final_biases = model.intercepts_

# Print them out!
print("Weights:")
for j, w in enumerate(final_weights):
    print(f"Layer {j}:\n{w}")

print("Biases:")
for j, b in enumerate(final_biases):
    print(f"Layer {j}:\n{b}")
