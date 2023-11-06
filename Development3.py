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
df = all_df.iloc[:300, :10]
column_names = df.columns.to_list()
sensors = column_names[3:]


fig = px.line(df, x="sample", y="xmeas_1")
fig.update_layout(dragmode=False)

myGraph = dcc.Graph(figure=fig)

mytext = dcc.Markdown(children="Hello!")

xAxis = dcc.Dropdown(
    options=sensors,
    value=sensors[0],
    style={'display': 'block'}  # Initially visible
)
yAxis = dcc.Dropdown(
    options=sensors,
    value=sensors[1],
    style={'display': 'block'}  # Initially visible
)
zAxis = dcc.Dropdown(
    options=sensors,
    value=sensors[2],
    style={'display': 'block'}  # Initially visible
)

checkbox = dcc.Checklist(
    options=sensors,
    value=[],
    style={'display': 'block'}
)

Button_SwitchView = html.Button(children='Switch to 3D')
Button_SwitchPlotType = html.Button(children='Switch to scatter')


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [Button_SwitchView, Button_SwitchPlotType, mytext, myGraph, checkbox, xAxis, yAxis, zAxis])


@app.callback(
    Output(mytext, 'children'),
    Output(myGraph, 'figure'),
    Output(Button_SwitchView, 'children'),
    Output(xAxis, 'style'),
    Output(yAxis, 'style'),
    Output(zAxis, 'style'),
    Output(checkbox, 'style'),
    Output(myGraph, 'style'),
    Output(Button_SwitchPlotType, 'style'),
    Output(Button_SwitchPlotType, 'children'),
    Input(checkbox, 'value'),
    Input(Button_SwitchView, 'n_clicks'),
    Input(xAxis, 'value'),
    Input(yAxis, 'value'),
    Input(zAxis, 'value'),
    Input(Button_SwitchPlotType, 'n_clicks')
)
def updateTimeGraph(selected_values, clicks, xAxis, yAxis, zAxis, scatterLineClicks):
    scatterLineText = ""
    if (clicks is None):
        clicks = 2
    if (scatterLineClicks is None):
        scatterLineClicks = 2

    if (clicks % 2 == 0):
        fig = px.line(df, x='sample',
                      y=selected_values, title='Time Based')

        buttonMessage = 'Switch to 3D'
        xAxisDisplay = {'display': 'none'}
        yAxisDisplay = {'display': 'none'}
        zAxisDisplay = {'display': 'none'}
        checkboxStyle = {'display': 'block'}
        scatterLineButtonStyle = {'display': 'none'}
        graphStyle = {'height': 400}
    if (clicks % 2 == 1):
        if (scatterLineClicks % 2 == 0):
            fig = px.line_3d(
                df, x=xAxis, y=yAxis, z=zAxis)
            scatterLineText = "Switch to scatter"
        if (scatterLineClicks % 2 == 1):
            fig = px.scatter_3d(
                df, x=xAxis, y=yAxis, z=zAxis)
            scatterLineText = "Switch to line"
        buttonMessage = 'Switch to time'
        xAxisDisplay = {'display': 'block'}
        yAxisDisplay = {'display': 'block'}
        zAxisDisplay = {'display': 'block'}
        checkboxStyle = {'display': 'none'}
        scatterLineButtonStyle = {'display': 'block'}
        graphStyle = {'height': 800}
    fig.update_layout(dragmode=False)
    return f'Selected values: {", ".join(selected_values)}', fig, buttonMessage, xAxisDisplay, yAxisDisplay, zAxisDisplay, checkboxStyle, graphStyle, scatterLineButtonStyle, scatterLineText


if __name__ == '__main__':
    app.run_server(debug=True)