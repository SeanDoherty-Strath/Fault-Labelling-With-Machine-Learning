
# OBJECTIVE
# The aim of this file is to create a website which lays out these steps:
# 1) Display a self-created, clear-featured data set
# 2) Perform K means on the data
# 3) Trace those clusters back to the time doman and check if 'faults' were identified

# 4) Perform PCA on the data set
# 5) Perform DBSCAN and plot the PCA grph
# 5) Trace back to the time domain and check if 'faults' were identified


import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd

# Read raw data
df = pd.read_csv(
    "RawData.csv"
)

fig_timeGraph = px.line(df, x='Time', y='sens1')
fig_timeGraph.update_yaxes(title_text="Sens 1", secondary_y=True)

fig_timeGraph.add_trace(px.line(df, x='Time', y='sens2').data[0])
fig_timeGraph.update_yaxes(title_text="Sens 2", secondary_y=True)


fig_timeGraph.add_trace(px.line(df, x='Time', y='sens3').data[0])
fig_timeGraph.add_trace(px.line(df, x='Time', y='sens4').data[0])


timeGraph = dcc.Graph(figure=fig_timeGraph)


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        dcc.Markdown('Hello!'),
        timeGraph
    ]
)

if __name__ == "__main__":
    app.run_server(debug=True)
