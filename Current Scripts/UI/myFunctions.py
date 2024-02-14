import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.metrics import silhouette_score
import math


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


def findBestParams(data, Range):
    # Define a range of epsilon values and min_samples values to search

    # eps_range = np.arange(1, 40, 5)
    eps_range = range(20, 50, 5)
    min_samples_range = range(3)

    best_score = -1
    best_eps = None
    best_min_samples = None

    # Perform grid search
    for eps_value in eps_range:
        for min_samples_value in min_samples_range:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            cluster_labels = dbscan.fit_predict(data)

            print('Eps: ', eps_value)
            print('Min: ', min_samples_value)
            numberOfLabels = len(list(set(cluster_labels)))

            print('Number of labels: ', numberOfLabels)
            if (numberOfLabels != 1):

                # Compute silhouette score
                score = silhouette_score(data, cluster_labels)
                print('Score: ', score)

            # Update best score and parameters if necessary
                minimumClusterSize = True
                myList = []
                # for i in range(min(cluster_labels), 1+max(cluster_labels)):
                #     myList = cluster_labels.tolist()
                #     if myList.count(i) < 20:
                #         minimumClusterSize = False
                if score > best_score and numberOfLabels >= 4 and numberOfLabels < 10 and minimumClusterSize:
                    # if score > best_score:
                    best_score = score
                    best_eps = eps_value
                    best_min_samples = min_samples_value

    print("Best silhouette score:", best_score)
    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)
    return best_eps, best_min_samples

    print("Best silhouette score:", best_score)
    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)


def performDBSCAN(data, eps, minVal):
    # Perform DBSCAN clustering

    # short_df = data.iloc[::3]
    print('Performing DBSCAN')
    # Range = data.values.max() - data.values.min()

    # best_eps, best_min_samples = findBestParams(data, Range)

    dbscan = DBSCAN(eps=eps, min_samples=minVal)
    dbscan.fit(data)
    labels = dbscan.labels_.tolist()

    # Add cluster labels to the dataset
    print('Number of labels: ', len(set(labels)))
    print('Min: ', min(set(labels)))
    return labels
