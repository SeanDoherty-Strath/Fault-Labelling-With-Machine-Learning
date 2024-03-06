import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd


df = pd.read_csv(
    "Data/TenesseeEastemen_FaultyTraining_Subsection.csv"
)
df = df.iloc[:, 3:]


column_names = df.columns.to_list()
print(column_names)
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


colours = [['blue'], ['green'], ['orange'], [
    'purple'], ['pink'], ['violet'], ['lavender']]


fig = px.scatter_3d(df_clusters, x=0, y=1, color_discrete_sequence=['black'])
for label in range(k):
    fig.add_trace(px.scatter_3d(cluster_dataframes[label], x=column_names[0], y=column_names[1], z=column_names[2],
                                color_discrete_sequence=colours[label]).data[0])


myGraph = dcc.Graph(figure=fig)

'''
fig = px.line(df, x="sample", y="xmeas_1")
fig.update_layout(dragmode=False)
myGraph = dcc.Graph(figure=fig)
'''

# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        myGraph
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
