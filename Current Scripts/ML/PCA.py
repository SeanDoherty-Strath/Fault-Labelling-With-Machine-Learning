import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pyreadr
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


df = pd.read_csv("TenesseeEastemen_FaultyTraining_Subsection.csv")
df = df.iloc[:, 3:6]

# To INFER the number of important components:
# pca = PCA(n_components="mle", svd_solver="full")

# To SET the number of components:
pca = PCA(n_components=3)
pca.fit(df)
df_PCA = pca.transform(df)

print('Raw data: ', df)

print("N featuers before: ", pca.n_features_in_)
print("N components after: ", pca.n_components_)
print("Explained variance ratio", pca.explained_variance_ratio_)
print(pca.singular_values_)

fig1 = px.scatter_3d(df, x='xmeas_1', y='xmeas_2',
                     z='xmeas_3', color_discrete_sequence=['red'])
myGraph_Raw = dcc.Graph(figure=fig1)

fig2 = px.scatter_3d(df_PCA, x=0, y=1, z=2, color_discrete_sequence=['blue'])
myGraph_PCA = dcc.Graph(figure=fig2)


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        html.H1("Raw Data"),
        myGraph_Raw,
        html.H1("PCA Data"),
        myGraph_PCA
    ]
)

print('PCA point 10: ', df_PCA[10])
print('Corresponds to raw data point 10: ',  df.iloc[9])

if __name__ == "__main__":
    app.run_server(debug=True)
