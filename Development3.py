from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import pyreadr
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

rdata_read = pyreadr.read_r(
    'D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData')

all_df = rdata_read['faulty_training']
df = all_df.iloc[:100, :10]
column_names = df.columns.to_list()
sensors = column_names[3:]


fig = px.line(df, x="sample", y="xmeas_1")
myGraph = dcc.Graph(figure=fig)

mytext = dcc.Markdown(children="Hello!")


checkbox = dcc.Checklist(
    options=sensors,
    value=[]
)


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container([mytext, myGraph, checkbox])


@app.callback(
    Output(mytext, 'children'),
    Output(myGraph, 'figure'),
    Input(checkbox, 'value')
)
def update_output(selected_values):

    # fig = px.line(df, x="sample", y=selected_values)

    fig = px.line(df, x='sample',
                  y=selected_values[0], title='Multiple Y-Axis Graph')
    fig.add_trace(
        px.line(df, x='sample', y=selected_values[1]).update_traces(yaxis="y2"))

    fig.update_layout(
        xaxis_title='X-Axis Title',
        yaxis_title='Y-Axis 1 Title',
        yaxis2=dict(title='Y-Axis 2 Title', overlaying='y', side='right')
    )

    return f'Selected values: {", ".join(selected_values)}', fig


# Run app
if __name__ == "__main__":
    app.run_server(port=8053)
