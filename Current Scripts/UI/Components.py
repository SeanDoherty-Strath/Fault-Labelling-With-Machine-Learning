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
mainGraph = dcc.Graph(figure=fig, style={'flex': 1})
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
    is_open=True,
    duration=4000,
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
sensorDropdown = dcc.Checklist(options=data.columns, value=['xmeas_1'], style={
                               'fontsize': 18, 'fontFamily': 'Comic Sans MS'}, inline=True, labelStyle={'width': '50%'})
sensorHeader = dcc.Markdown('Sensors', style={
    'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
xAxis_dropdown_3D = dcc.Dropdown(value=data.columns[0], options=data.columns)
yAxis_dropdown_3D = dcc.Dropdown(value=data.columns[1], options=data.columns)
zAxis_dropdown_3D = dcc.Dropdown(value=data.columns[2], options=data.columns)

# Box 2
zoom = dcc.Slider(0, 10, 1, value=0)
pan = dcc.Slider(0, 2000, 100, value=0)


faultFinderHeader = html.H3("Fault Finder", style={'margin': '0'})

faultFinderText = html.P('Find next:')
faultFinder = dcc.Dropdown(value='Unlabelled Data Point', options=[
                           'Unlabelled Data Point', 'No Fault', 'Fault 1', 'Fault 2', 'Fault 3'])


# Box 4
labelDropdown = dcc.Dropdown(value='No Fault', options=[
                             'Fault 1', 'Fault 2', 'Fault 3', 'No Fault'])


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

stat1 = dcc.Markdown('No. Labels Placed: ', style={'margin': '0'})
stat2 = dcc.Markdown('No. Types Labels Placed: ', style={'margin': '0'})
stat3 = dcc.Markdown('Data points unlabelled: ', style={'margin': '0'})


AI_header = dcc.Markdown('Auto Label', style={
                         'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
AI_text1 = dcc.Markdown('Clustering Method:', style={
                        'width': '50%', 'margin-left': 10, })
AI_input1 = dcc.Input(id='AI-input1', type='number', value='K')
AI_input2 = dcc.Input(id='AI-input2', type='number', value='Empty')
clusterMethod = dcc.Dropdown(
    options=['K Means', 'DBSCAN'], value='K Means', style={'width': '100%'})
AI_text2 = dcc.Markdown('Reduction Algorithm:')
reductionMethod = dcc.Dropdown(
    options=["PCA", "Autoencoding", "None"], value='None', style={'width': '100%'})
AI_input3 = dcc.Input(id='AI-input1', type='number',)
AI_input4 = dcc.Input(id='AI-input2', type='number',)
AI_checkbox = dcc.Checklist(id='AI-checkbox', options=data.columns, )


exportHeader = html.H3('Export to CSV')

exportName = dcc.Input(
    id='file-name',
    type='text',
    value='Filename',
    style={'width': '100%', 'height': 30, 'fontSize': 20
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
