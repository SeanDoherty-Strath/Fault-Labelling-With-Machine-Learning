import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score


def changeText(text):
    return text


def updateGraph(value, data):
    fig = px.line(data, y=value)


sensors = []
for i in range(52):
    temp = 'Sensor' + str(i)
    sensors.append(temp)
    # print(sensors)

print(sensors)


def performPCA(df, n):
    print('Perform PCA')
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df)

    pca = PCA(n_components=n)
    principal_components = pca.fit_transform(scaled_data)

    columns = []
    for i in range(n):

        columns.append('PCA' + str(i))
    principal_df = pd.DataFrame(
        data=principal_components, columns=columns)
    return principal_df


def performKMeans(df, k):

    # Create a KMeans instance
    kmeans = KMeans(n_clusters=k, n_init="auto")

    # Fit the model to the data
    kmeans.fit(df)

    # Get the cluster labels and centroids
    labelArray = kmeans.labels_
    labels = labelArray.tolist()

    return labels


def findBestParams(data):
    # Define a range of epsilon values and min_samples values to search

    eps_range = np.arange(20, 35, 1)
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

            if (len(list(set(cluster_labels))) != 1):

                # Compute silhouette score
                score = silhouette_score(data, cluster_labels)
                print(score)

            # Update best score and parameters if necessary
                if score > best_score:
                    best_score = score
                    best_eps = eps_value
                    best_min_samples = min_samples_value
    return best_eps, best_min_samples

    print("Best silhouette score:", best_score)
    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)


def performDBSCAN(data):
    # Perform DBSCAN clustering

    short_df = data.iloc[::3]

    best_eps, best_min_samples = findBestParams(short_df)

    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan.fit(data)
    labels = dbscan.labels_.tolist()

    # Add cluster labels to the dataset
    print('Number of labels: ', len(set(labels)))
    print('Min: ', min(set(labels)))
    return labels
