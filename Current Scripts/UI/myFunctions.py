import plotly.express as px
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


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
    # centroids = kmeans.cluster_centers_
    # df_clusters = pd.DataFrame(centroids)
    # cluster_dataframes = []

    # print(df_clusters)

    # for label in range(k):
    # #     # Filter data based on cluster label
    #     cluster_df = df[labels == label]
    #     cluster_dataframes.append(cluster_df)

    # colours = [['blue'], ['green'], ['orange'], [
    #     'purple'], ['pink'], ['violet'], ['lavender']]

    # fig = px.scatter_3d(df_clusters, x=0, y=1, z=2, color_discrete_sequence=['black'])
    # for label in range(k):
    #     fig.add_trace(px.scatter_3d(cluster_dataframes[label], x=df.columns.to_list()[0], y=df.columns.to_list()[1], z=df.columns.to_list()[2],
    #                                 color_discrete_sequence=colours[label]).data[0])

    # return fig
