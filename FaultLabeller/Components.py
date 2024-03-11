import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc

# DATA
data = pd.read_csv("FaultLabeller/Data/UpdatedData.csv")
data = data.iloc[:, 4:]

# Top
title = html.H1(children='Fault Labeller', style={
                'fontsize': 100, 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'left', 'fontFamily': ''})
mainGraph = dcc.Graph(figure=px.line(), style={'flex': '1'})


# Box 1
sensorHeader = dcc.Markdown('Sensors', style={
    'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
sensorDropdown = dcc.Checklist(options=data.columns, value=[data.columns[0]], style={
                               'fontSize': 20, 'margin': 10}, inline=True, labelStyle={'width': '33%'})
sensorText = dcc.Markdown(
    'Select which sensors you wish to see in the graph above.', style={'textAlign': 'center', 'fontSize': 18})

xAxis_dropdown_3D = dcc.Dropdown(
    value=data.columns[0], options=data.columns, style={'width': '50%'})
yAxis_dropdown_3D = dcc.Dropdown(
    value=data.columns[1], options=data.columns, style={'width': '50%'})
zAxis_dropdown_3D = dcc.Dropdown(
    value=data.columns[2], options=data.columns, style={'width': '50%'})

xAxisText = dcc.Markdown(
    'x axis: ', style={'margin-left': 50, 'fontSize': 20, 'width': '50%'})
yAxisText = dcc.Markdown(
    'y axis: ', style={'margin-left': 50, 'fontSize': 20, 'width': '50%'})
zAxisText = dcc.Markdown(
    'z axis: ', style={'margin-left': 50, 'fontSize': 20, 'width': '50%'})


# Box 3

faultFinderHeader = html.H3("Fault Finder", style={'margin': '0'})

faultFinderText = dcc.Markdown(
    'Search for:', style={'fontSize': 20, 'width': '25%'})
faultFinder = dcc.Dropdown(value='Unlabelled Data Point', options=[
                           'Unlabelled Data Points', 'No Fault', 'Fault 1', 'Fault 2', 'Fault 3'], style={'width': '90%'})


# Box 2
labelDropdown = dcc.Dropdown(value=1, options=[
    {'label': 'Unlabelled', 'value': 0},
    {'label': 'No Fault (Green)', 'value': 1},
    {'label': 'Fault 1 (Red)', 'value': 2},
    {'label': 'Fault 2 (Orange)', 'value': 3},
    {'label': 'Fault 3 (Yellow)', 'value': 4},
    {'label': 'Fault 4 (Pink)', 'value': 5},
    {'label': 'Fault 5 (Purple)', 'value': 6},
    {'label': 'Fault 6 (Lavender)', 'value': 7},
    {'label': 'Fault 7 (Blue)', 'value': 8},
    {'label': 'Fault 8 (Brown)', 'value': 9},
    {'label': 'Fault 9 (Cyan)', 'value': 10}],
    style={'height': '20%'})


# Box 4
AI_header = dcc.Markdown('Automatically Label Faults', style={
                         'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
clusterMethod = dcc.Dropdown(
    options=['K Means', 'DBSCAN', 'Neural Network (Supervised)'], value='K Means', style={'width': '100%'})
reductionMethod = dcc.Dropdown(
    options=["PCA", "Auto-encoding", "None"], value='None', style={'width': '100%'})


# Box 5
stat1 = dcc.Markdown('No. Labels Placed: ', style={
                     'margin-left': 10, 'fontSize': 20})
stat2 = dcc.Markdown('No. Types Labels Placed: ', style={
                     'display': 'none'})
stat3 = dcc.Markdown('Data points unlabelled: ', style={
                     'margin-left': 10, 'fontSize': 20})


# Box 6
uploadData = dcc.Upload(

    children=html.Div([
        'Drag and Drop or ',
        html.A('Select Files'),
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },
    # Allow multiple files to be uploaded
    multiple=False
),


exportName = dcc.Input(
    id='LabelledData',
    type='text',
    value='Filename',
    style={'width': '70%', 'height': 30, 'fontSize': 20
           }
)


exportConfirm = html.Button(
    'Export',
    style={'width': '20%', 'height': 30}
)
