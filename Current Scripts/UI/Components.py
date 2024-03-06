import dash_core_components as dcc
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc

# DATA
data = pd.read_csv("Data/UpdatedData.csv")
data = data.iloc[:, 4:]

# Top
title = html.H1(children='Fault Labeller', style={
                'fontsize': 100, 'color': 'white', 'fontWeight': 'bold', 'textAlign': 'left', 'fontFamily': ''})

# Main Graph
fig = px.line()
fig.update_layout()
mainGraph = dcc.Graph(figure=fig, style={'flex': '1'})
# mainGraph = dcc.Graph(figure=fig, config={'editable': False, 'edits': {'shapePosition': True}},)


# alert = dbc.Modal(
#      is_open=True,
#     #  style={"position": "fixed"}
#      children=
#         [
#             dbc.ModalHeader("Example Modal Header"),
#             dbc.ModalBody("This is the content of the modal"),
#             dbc.ModalFooter(
#                 # dbc.Button("Close", id="close-modal-button", className="ml-auto")
#             ),
#         ],
#     )

# alert = html.Div(
#     style={
#         # top': '20px',
#         'width':'300px',
#         'height':'50px',
#         'justify-content':'center',
#         'align-items':'center',
#         'text-align': 'center',
#         'margin': '10px',
#     },
#     children=
#     [
#         dbc.Alert(
#             "Hey asshole!",
#             # id="alert-auto",
#             # is_open=True,
#             duration=4000,
#             color="lightblue",
#             style={
#                 'font-size': 20,
#             }
#         )]
# )

alert = dbc.Alert(
    "WARNING: This is not ok",
    is_open=False,
    # duration=4000,
    color="lightblue",
    style={
        'font-size': 30,
        'font-color': 'white',
        # top': '20px',
        'padding': '20px',
        'height': '50px',
        'justify-content': 'center',
        'align-items': 'center',
        'text-align': 'center',
        'align-self': 'center',
        'position': 'absolute',

    }
)

# Box 1
sensorDropdown = dcc.Checklist(options=data.columns, value=[data.columns[0]], style={
                               'fontSize': 20, 'margin': 10}, inline=True, labelStyle={'width': '33%'})
sensorText = dcc.Markdown(
    'Select which sensors you wish to see in the graph above.', style={'textAlign': 'center', 'fontSize': 18})
sensorHeader = dcc.Markdown('Sensors', style={
    'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
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


# Box 2
zoom = dcc.Slider(0, 10, 1, value=0)
pan = dcc.Slider(0, 2000, 100, value=0)


faultFinderHeader = html.H3("Fault Finder", style={'margin': '0'})

faultFinderText = dcc.Markdown(
    'Search for:', style={'fontSize': 20, 'width': '25%'})
faultFinder = dcc.Dropdown(value='Unlabelled Data Point', options=[
                           'Unlabelled Data Points', 'No Fault', 'Fault 1', 'Fault 2', 'Fault 3'], style={'width': '90%'})


# Box 4
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


# Multi-Graph View
fourGraphs = html.Div(style={'margin': '0', 'padding': '0'}, children=[
    # First Row
    html.Div(style={'margin': '0', 'padding': '0'}, children=[
        dcc.Graph(
            id='subplot-graph-1',
            figure={
                'data': [
                    go.Line(
                        # x=[1, 2, 3],
                        # y = [4, 6, 7],
                        y=data.iloc[:, 2]
                    ),
                ],
                'layout': go.Layout(height=350)},
            config={'displayModeBar': False}

        ),
        dcc.Graph(
            id='subplot-graph-1',
            figure={
                'data': [
                    go.Line(
                        y=data.iloc[:, 3]

                    ),
                ],
                'layout': go.Layout(height=350)},
            config={'displayModeBar': False}
        ),
        dcc.Graph(
            id='subplot-graph-1',
            figure={
                'data': [
                    go.Line(

                        y=data.iloc[:, 4]

                    ),
                ],
                'layout': go.Layout(height=350)},
            config={'displayModeBar': False}
        ),
    ])])

# Box 5
stats = html.Div(style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'left', 'align-items': 'center'},
                 children=[
    html.H3("Stats", style={'margin': '1'}),
    #  html.P('No. Labels placed: ', style={'margin':'0'},),
    #  html.P('Nof. Types Labels placed: ', style={'margin':'0'}),

]
)

stat1 = dcc.Markdown('No. Labels Placed: ', style={
                     'margin-left': 10, 'fontSize': 20})
stat2 = dcc.Markdown('No. Types Labels Placed: ', style={
                     'display': 'none'})
stat3 = dcc.Markdown('Data points unlabelled: ', style={
                     'margin-left': 10, 'fontSize': 20})


AI_header = dcc.Markdown('Automatically Label Faults', style={
                         'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
AI_text1 = dcc.Markdown('Clustering Method:', style={
                        'width': '50%', 'margin-left': 10, })
AI_input1 = dcc.Input(id='AI-input1', type='number', value='K')
AI_input2 = dcc.Input(id='AI-input2', type='number', value='Empty')
clusterMethod = dcc.Dropdown(
    options=['K Means', 'DBSCAN'], value='K Means', style={'width': '100%'})
AI_text2 = dcc.Markdown('Reduction Algorithm:')
reductionMethod = dcc.Dropdown(
    options=["PCA", "Auto-encoding", "None"], value='None', style={'width': '100%'})
AI_input3 = dcc.Input(id='AI-input1', type='number',)
AI_input4 = dcc.Input(id='AI-input2', type='number',)
AI_checkbox = dcc.Checklist(id='AI-checkbox', options=data.columns, )

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

exportHeader = html.H3('Import / Export Data')

exportName = dcc.Input(
    id='LabelledData',
    type='text',
    value='Filename',
    style={'width': '70%', 'height': 30, 'fontSize': 20
           }
)

exportLocation = dcc.Input(
    id='file-location',
    type='text',
    value='File Location',
    style={}
)
exportConfirm = html.Button(
    'Export',
    style={'width': '20%', 'height': 30}
)
