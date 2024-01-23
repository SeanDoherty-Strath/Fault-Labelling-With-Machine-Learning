
# NOTE
# This file, datalabelling 1, has the most features but it badly designed
# Datalabelling 2 was being laid out better, till Robert changed my task.
# When you return to it, build up databalleing 2
# Databalleing 3 was left becasue it has some desiarble features

from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objs as go

df = pd.read_csv("Data/UpdatedData.csv")
column_names = df.columns.to_list()
sensors = column_names[4:]

fig = px.line(df, x="sample", y="xmeas_1")
fig.update_layout(dragmode=False)

myGraph = dcc.Graph(figure=fig)

mytext = dcc.Markdown(children="Hello!")

xAxis = dcc.Dropdown(
    options=sensors, value=sensors[0], style={
        "display": "block"}  # Initially visible
)
yAxis = dcc.Dropdown(
    options=sensors, value=sensors[1], style={
        "display": "block"}  # Initially visible
)
zAxis = dcc.Dropdown(
    options=sensors, value=sensors[2], style={
        "display": "block"}  # Initially visible
)

slider = dcc.RangeSlider(
    min=0,
    max=500,
    step=1,
    marks={i: str(i) for i in range(500)},
    value=[0, 500 - 1],
)

checkbox = dcc.Dropdown(options=sensors, value='xmeas_1',
                        style={"display": "block"})

Button_SwitchView = html.Button(children="Switch to 3D")
Button_SwitchPlotType = html.Button(children="Switch to scatter")


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        Button_SwitchView,
        Button_SwitchPlotType,
        mytext,
        myGraph,
        slider,
        checkbox,
        xAxis,
        yAxis,
        zAxis,
    ]
)


@app.callback(
    Output(mytext, "children"),
    Output(myGraph, "figure"),
    Output(Button_SwitchView, "children"),
    Output(xAxis, "style"),
    Output(yAxis, "style"),
    Output(zAxis, "style"),
    Output(checkbox, "style"),
    Output(myGraph, "style"),
    Output(Button_SwitchPlotType, "style"),
    Output(Button_SwitchPlotType, "children"),
    Input(checkbox, "value"),
    Input(Button_SwitchView, "n_clicks"),
    Input(xAxis, "value"),
    Input(yAxis, "value"),
    Input(zAxis, "value"),
    Input(Button_SwitchPlotType, "n_clicks"),
    Input(slider, 'value')
)
def updateTimeGraph(selected_values, clicks, xAxis, yAxis, zAxis, scatterLineClicks, selected_range):
    scatterLineText = ""
    if clicks is None:
        clicks = 2
    if scatterLineClicks is None:
        scatterLineClicks = 2

    start, end = selected_range

    x = df['sample']
    x_range = x[start:end+1]
    y = df[selected_values]
    y_range = y[start:end+1]

    print(x_range)
    print(y_range)
    print('Selected Values: ', selected_values)

    if clicks % 2 == 0:
        fig = px.line(df, x="sample", y=selected_values)
        fig = {
            'data': [{'x': x_range, 'y': y_range, 'type': 'scatter'}],
            'layout': {
                'xaxis': {'range': [x[start], x[end]]},
                'yaxis': {'title': 'Value'},
                'title': 'Time Based Graph'
            }
        }
        buttonMessage = "Switch to 3D"
        xAxisDisplay = {"display": "none"}
        yAxisDisplay = {"display": "none"}
        zAxisDisplay = {"display": "none"}
        checkboxStyle = {"display": "block"}
        scatterLineButtonStyle = {"display": "none"}
        graphStyle = {"height": 400}
        # fig.update_layout(dragmode=False)
    if clicks % 2 == 1:
        if scatterLineClicks % 2 == 0:
            fig = px.line_3d(df, x=xAxis, y=yAxis, z=zAxis)
            scatterLineText = "Switch to scatter"
        if scatterLineClicks % 2 == 1:
            fig = px.scatter_3d(df, x=xAxis, y=yAxis, z=zAxis)
            scatterLineText = "Switch to line"
        buttonMessage = "Switch to time"
        xAxisDisplay = {"display": "block"}
        yAxisDisplay = {"display": "block"}
        zAxisDisplay = {"display": "block"}
        checkboxStyle = {"display": "none"}
        scatterLineButtonStyle = {"display": "block"}
        graphStyle = {"height": 400}
        

    return (
        f'Selected values: {", ".join(selected_values)}',
        fig,
        buttonMessage,
        xAxisDisplay,
        yAxisDisplay,
        zAxisDisplay,
        checkboxStyle,
        graphStyle,
        scatterLineButtonStyle,
        scatterLineText,
    )


if __name__ == "__main__":
    app.run_server(debug=True)
