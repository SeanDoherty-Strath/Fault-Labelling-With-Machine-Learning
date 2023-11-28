import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd
from pathlib import Path


# Generate some sample data for clustering

df = pd.read_csv(
    "LatentSpace.csv"
)

# column_names = df.columns.to_list()
# sensors = column_names[3:]

# Define the number of clusters (K)
k = 3

# Create a KMeans instance
kmeans = KMeans(n_clusters=k, n_init="auto")

# Fit the model to the data
kmeans.fit(df)

# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
df_clusters = pd.DataFrame(centroids)
cluster_dataframes = []
for label in range(k):
    # Filter data based on cluster label
    cluster_df = df[labels == label]
    cluster_dataframes.append(cluster_df)

trend = '2'

print(cluster_dataframes[0])
fig = px.line(cluster_dataframes[0], x='Unnamed: 0',
              y=trend, color_discrete_sequence=['aqua'])
fig.add_trace(px.line(cluster_dataframes[1], x='Unnamed: 0',
              y=trend, color_discrete_sequence=['red']).data[0])
fig.add_trace(px.line(cluster_dataframes[2], x='Unnamed: 0',
              y=trend, color_discrete_sequence=['green']).data[0])
myGraph = dcc.Graph(figure=fig)


fig2 = px.scatter_3d(
    cluster_dataframes[0], x='1', y='2', z='3', color_discrete_sequence=['aqua'])
fig2.add_trace(px.scatter_3d(
    cluster_dataframes[1], x='1', y='2', z='3', color_discrete_sequence=['red']).data[0])
fig2.add_trace(px.scatter_3d(
    cluster_dataframes[2], x='1', y='2', z='3', color_discrete_sequence=['green']).data[0])

myGraph2 = dcc.Graph(figure=fig2)

# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        myGraph,
        myGraph2
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
