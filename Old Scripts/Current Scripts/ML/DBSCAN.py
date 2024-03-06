import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import pandas as pd


data = pd.read_csv("Data/UpdatedData.csv")
df = data.drop(data.columns[[0, 1, 2, 3]], axis=1)  # Remove extra columns
# df = StandardScaler().fit_transform(data)

# Calculate the k-distance graph
min = 5
neigh = NearestNeighbors(n_neighbors=min)
nbrs = neigh.fit(df)
distances, indices = nbrs.kneighbors(df)
distances = np.sort(distances, axis=0)
distances = distances[:, 1]
knee_point = np.argmax(np.diff(distances, 2)) + 2
optimal_eps = distances[knee_point]

print(distances)

print('Optimal Knee Point', knee_point)
print("Optimal Epsilon:", optimal_eps)


# Plot the k-distance graph
plt.plot(distances, 'red')
plt.xlabel("Data Points")
plt.ylabel("Epsilon")
plt.title("K-distance Graph")
plt.show()


# NOTE: I am sceptical about this code. It always producing the same value (0.69)
# I took it from a tutorial which explains it:
# np.diff(distances, 2): This computes the second-order differences of the distances array.It is a way to identify points where the rate of change is changing i.e. a knee point
#  np.argmax(...): returns the indices of the max values range of change
#  + 2: This is added to the result because the np.diff operation reduces the size of the array by 1. The knee point is then adjusted to align with the original data points.

# I think a better solution wil be to eps from eyeseight... although this mght be hard to scale


# Now, use the optimal_eps in DBSCAN
dbscan = DBSCAN(eps=optimal_eps, min_samples=min)
labels = dbscan.fit_predict(df)

# Visualize the clusters
plt.scatter(df[:, 0], df[:, 1], c=labels, cmap="viridis")
plt.title("DBSCAN Clustering with Optimal Epsilon")
plt.show()
