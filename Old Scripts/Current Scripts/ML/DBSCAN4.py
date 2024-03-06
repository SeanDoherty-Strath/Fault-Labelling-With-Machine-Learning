import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from sklearn.preprocessing import StandardScaler


def knee_point(X, k):
    # Fit a k-nearest neighbor model
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)

    # Compute distances to k-nearest neighbors
    distances, _ = nn.kneighbors(X)
    avg_distances = np.mean(distances, axis=1)

    # Sort distances in ascending order
    sorted_distances = np.sort(avg_distances)

    # Calculate the cumulative distribution function
    cdf = np.cumsum(sorted_distances)
    cdf /= cdf[-1]

    # Find the knee point
    knee_point_index = np.argmax(cdf >= 0.9)
    knee_point_value = sorted_distances[knee_point_index]

    return knee_point_value


# Example usage
# X is your data
# k is the number of nearest neighbors to consider
data = pd.read_csv("Data/UpdatedDataFebruary24.csv")
X = data.drop(data.columns[[0, 1, 2, 3]], axis=1)  # Remove extra columns
# X = StandardScaler().fit_transform(data)

# df = StandardScaler().fit_transform(data)
# X = np.random.rand(100, 2)  # Example random 2D data
k = 52  # Choose the number of nearest neighbors

# Calculate the knee point
epsilon = knee_point(X, k+1)
print('epsilon: ', epsilon)

# Plot the sorted distances
sorted_distances = np.sort(np.mean(NearestNeighbors(
    n_neighbors=k).fit(X).kneighbors()[0], axis=1))
plt.plot(sorted_distances)
plt.xlabel('Points')
plt.ylabel('Distance to {}th Nearest Neighbor'.format(k))
plt.title('Knee Point Method')
plt.grid(True)
plt.show()

print("Optimal epsilon:", epsilon)
