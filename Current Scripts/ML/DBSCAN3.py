import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt


def findBestParams(data):
    # Define a range of epsilon values and min_samples values to search

    eps_range = np.arange(30, 50, 5)
    min_samples_range = range(1, 5)

    best_score = -1
    best_eps = None
    best_min_samples = None

    # Perform grid search
    for eps_value in eps_range:
        for min_samples_value in min_samples_range:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            cluster_labels = dbscan.fit_predict(data)

            numberOfLabels = len(list(set(cluster_labels)))
            print('Number of labels: ', numberOfLabels)

            if (len(list(set(cluster_labels))) != 1):

                # Compute silhouette score
                score = silhouette_score(data, cluster_labels)
                print(score)

            # Update best score and parameters if necessary
                if score > best_score and numberOfLabels >= 3 and numberOfLabels <= 10:
                    best_score = score
                    best_eps = eps_value
                    best_min_samples = min_samples_value
    print('Best EPS: ', best_eps)
    print('Best Min Samples: ', best_min_samples)
    return best_eps, best_min_samples


def performDBSCAN(data):
    # Perform DBSCAN clustering

    # short_df = data.iloc[::3]

    best_eps, best_min_samples = findBestParams(data)

    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan.fit(data)
    labels = dbscan.labels_.tolist()

    # Add cluster labels to the dataset
    print('Number of labels: ', len(set(labels)))
    print('Min: ', min(set(labels)))
    return labels


# DATA
data = pd.read_csv("Data/UpdatedData.csv")
data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)  # Remove extra columns

labels = performDBSCAN(data)
# Visualize the clusters
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap="viridis")
plt.title("DBSCAN Clustering with Optimal Epsilon")
plt.show()
