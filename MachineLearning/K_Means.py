import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd

# Generate some sample data for clustering
data = np.random.rand(100, 2)
df = pd.read_csv(
    "TenesseeEastemen_FaultyTraining_Subsection.csv"
)
df = df.iloc[:, 3:5]

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

print('Cluster Dataframes')
print(cluster_dataframes)

colours = [['blue'], ['green'], ['orange']]

fig = px.scatter(df_clusters, x=0, y=1, color_discrete_sequence=['red'])
for label in range(k):
    fig.add_trace(px.scatter(cluster_dataframes[label], x='xmeas_1', y='xmeas_2',
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
