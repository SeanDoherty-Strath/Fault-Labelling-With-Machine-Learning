import numpy as np
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd
import pyreadr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from pathlib import Path

# Read in raw data (300 with 7 sensors)
# df = pd.read_csv(
#     "TenesseeEastemen_FaultyTraining_Subsection.csv"
# )
# # Remove the first three columns (always the same for this dataset)
# df = df.iloc[:, 3:8]
# column_names = df.columns.to_list()

# rdata_read = pyreadr.read_r(
#     "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
# all_df = rdata_read["faulty_training"]
# df = all_df.iloc[:1000, 3:10]
# column_names = df.columns.to_list()

df = pd.read_csv(
    "LatentSpace.csv"
)
# Remove the first three columns (always the same for this dataset)
column_names = df.columns.to_list()

# C0MPONENTS

cluster_dataframes = []

fig = {}
myGraph = dcc.Graph(figure=fig)

mytext0 = dcc.Markdown(children="Algorithm: ")
dropdown_algorithm = dcc.Dropdown(
    options=['K Means', 'DBSCAN', 'Other'],
    value='DBSCAN'
)

mytext1 = dcc.Markdown(children="Number of clutsters: ")
dropwdown_k = dcc.Dropdown(
    options=[1, 2, 3, 4, 5, 6],
    value=3,
)

mytext2 = dcc.Markdown(children="X axis")
dropwdown_x = dcc.Dropdown(
    options=column_names,
    value=column_names[0],
)
mytext3 = dcc.Markdown(children="Y axis")
dropwdown_y = dcc.Dropdown(
    options=column_names,
    value=column_names[1],
)
mytext4 = dcc.Markdown(children="Z axis")
dropwdown_z = dcc.Dropdown(
    options=column_names,
    value=column_names[2],
)


mytext6 = dcc.Markdown(children="Min Val")
slider_minVal = dcc.RangeSlider(
    min=0,
    max=20,
    step=1,
    marks={i: str(i) for i in range(20)},
    value=[5],
)


text = dcc.Markdown(children="")
# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        text, myGraph, mytext0, dropdown_algorithm, mytext6, slider_minVal, mytext1, dropwdown_k, mytext2, dropwdown_x, mytext3, dropwdown_y, mytext4, dropwdown_z, html.Button(
            title='Export CSV', id='myButton'),

    ]
)


@app.callback(
    Output(text, 'children'),
    Input('myButton', 'n_clicks')
)
def exportCSV(n_clicks):
    return n_clicks


@app.callback(
    Output(myGraph, "figure"),
    Input(dropwdown_k, 'value'),
    Input(dropwdown_x, 'value'),
    Input(dropwdown_y, 'value'),
    Input(dropwdown_z, 'value'),
    Input(dropdown_algorithm, 'value'),
    Input(slider_minVal, 'value')
)
def updatePlot(k, xAxis, yAxis, zAxis, algorithm,  min):
    min = min[0]
    colours = [['grey'], ['blue'], ['green'], ['orange'], [
        'purple'], ['pink'], ['violet'], ['lavender']]
    coloursContinuous = [['#FF0000'], ['#FF00B3'], ['#D500FF'], ['#7700FF'], ['#0900FF'], ['#00B3FF'], [
        '#00FFDE'], ['#00FF66'], ['#80FF00'], ['#EFFF00'], ['#FFB300'], ['#FF4400'], ['#FF0000']]
    if (algorithm == 'K Means'):
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
        print(df_clusters)

        fig = px.scatter_3d(df_clusters, x=0, y=1, z=2,
                            color_discrete_sequence=['black'])
        for label in range(0, k):
            fig.add_trace(px.scatter_3d(cluster_dataframes[label], x=xAxis, y=yAxis, z=zAxis,
                                        color_discrete_sequence=colours[label]).data[0])
    elif (algorithm == 'DBSCAN'):
        ''' get optimal eps using user-selected min val'''
        neigh = NearestNeighbors(n_neighbors=min)
        nbrs = neigh.fit(df)
        distances, indices = nbrs.kneighbors(df)
        distances = np.sort(distances, axis=0)
        distances = distances[:, 1]
        knee_point = np.argmax(np.diff(distances, 2)) + 2
        optimal_eps = distances[knee_point]

        clustering = DBSCAN(eps=optimal_eps, min_samples=min).fit(df)
        labels = clustering.labels_
        print('Optimal EPS for min val ', min, ' is ', optimal_eps)
        cluster_dataframes = []
        outliers_df = df[labels == -1]
        print('Max labels: ', max(labels))
        for label in range(max(labels)+1):
            print('Splitting label: ', label)
            # Filter data based on cluster label
            cluster_df = df[labels == label]
            cluster_dataframes.append(cluster_df)
        print(outliers_df)

        fig = px.scatter_3d(
            outliers_df, x=xAxis, y=yAxis, z=zAxis, color_discrete_sequence=['grey'])
        for label in range(max(labels)+1):
            print('Colouring Label: ', label)
            fig.add_trace(px.scatter_3d(cluster_dataframes[label], x=xAxis, y=yAxis, z=zAxis,
                                        color_discrete_sequence=coloursContinuous[label]).data[0])

        clustersDF = pd.DataFrame(cluster_dataframes)
        filepath = Path('./Clusters.csv')
        clustersDF.to_csv(filepath)

    else:
        fig = px.scatter_3d()

    return (
        fig
    )


if __name__ == "__main__":
    app.run_server(debug=True)
