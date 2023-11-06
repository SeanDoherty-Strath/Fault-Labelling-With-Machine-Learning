from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pyreadr
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# INITALISE APP
app = Dash(__name__, external_stylesheets=[dbc.themes.SOLAR])

# DATA HANDLING
# Get Data from Tenesee Eastmen
rdata_read = pyreadr.read_r(
    'D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData')
# Save all data in a panda dataframe
all_df = rdata_read['faulty_training']
# Cut down data to a manageable amount
df = all_df.iloc[:100, :10]
# Make a note of all the columns
column_names = df.columns.to_list()
# Make a note of all the sensors
sensors = column_names[3:]

# COMPONENTS AND LAYOUT
app.layout = html.Div([
    html.H1("Sample Dash App with Centered Graph"),

    html.Div([
        dcc.Graph(
            id='mainGraph',
            figure=px.line(df, x="sample", y="xmeas_3")
        ),
    ],
        style={'width': '80%', 'display': 'inline-block', 'padding': '10px', 'text-align': 'center', 'margin': '0 auto'}),  # Center the graph
    dcc.Checklist(
        options=sensors,
        value=[sensors[0]]),
], style={'display': 'flex', })


# RUN APP
if __name__ == '__main__':
    app.run_server(debug=True, threaded=True)
