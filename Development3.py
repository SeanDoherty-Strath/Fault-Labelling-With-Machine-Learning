from dash import Dash, dcc, Output, Input, html, State
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

xAxis = dcc.Dropdown(
    options=[
        'option1',  'option2'
    ],
    value='option1',
    style={'display': 'block'}  # Initially visible
)

checkbox = dcc.Checklist(
    options=sensors,
    value=[]
)

Button_SwitchView = html.Button(children='Switch to 3D')


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [Button_SwitchView, mytext, myGraph, checkbox, xAxis])


@app.callback(
    Output(mytext, 'children'),
    Output(myGraph, 'figure'),
    Output(Button_SwitchView, 'children'),
    Output(xAxis, 'style'),
    Input(checkbox, 'value'),
    Input(Button_SwitchView, 'n_clicks')
)
def updateTimeGraph(selected_values, clicks):
    if (clicks % 2 == 0):
        fig = px.line(df, x='sample',
                      y=selected_values, title='Time Based')
        buttonMessage = 'Switch to 3D'
        xAxisDisplay = {'display': 'none'}
    if (clicks % 2 == 1):
        print(selected_values[0])
        fig = px.scatter_3d(
            df, x=selected_values[0], y=selected_values[1], z=selected_values[2])
        buttonMessage = 'Switch to time'
        xAxisDisplay = {'display': 'block'}
    return f'Selected values: {", ".join(selected_values)}', fig, buttonMessage, xAxisDisplay


if __name__ == '__main__':
    app.run_server(debug=True)
