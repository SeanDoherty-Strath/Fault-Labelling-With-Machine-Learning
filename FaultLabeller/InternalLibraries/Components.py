import dash_core_components as dcc
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import html
import dash_bootstrap_components as dbc
import FaultLabeller.InternalLibraries.Styles as Styles

# Placeholder data 
data = pd.DataFrame({
    'Sensor 1': [],
    'Sensor 2': [],
    'Sensor 3': [],
    'labels': []
})

# Top box
title = dcc.Markdown(children='Upload Data to Begin Fault Labelling', style={
                     'color': 'white', 'fontSize': 30,  'padding-bottom': 15, 'position': 'absolute', 'top': 0})

topBox = html.Div(style=Styles.topBox,
                  children=[
                      html.Button('Switch View', id='switchView',
                                  style={'fontSize': 20, 'margin': 20, 'position': 'absolute', 'left': 0, 'top': 0}),
                      html.Button('View Time Representation', id='switchRepresentation', style={
                          'fontSize': 20, 'margin': 20, 'position': 'absolute', 'left': 130, 'top': 0, 'display': 'none'}),
                      html.Button("Open Comments", id="open-modal",
                                  style={
                                      'fontSize': 20, 'margin': 20, 'position': 'absolute', 'right': 0, 'top': 0}),
                      html.Div(style={'flex-direction': 'row', 'display': 'flex', 'width': '100%', 'height': '100%', },
                               children=[
                          html.Div(id='ClusterColourContainer', style={"display": "none", 'flex': '1'},
                                   children=[
                              dcc.Dropdown(
                                  style={'display': 'none'}, id='dropdown-0'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id='dropdown-1'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-2'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-3'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-4'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-5'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-6'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-7'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-8'),
                              dcc.Dropdown(
                                  style={'display': 'none'}, id=f'dropdown-9'),
                              html.Button(
                                  style={'display': 'none'}, id='colorNow'),]),
                          dcc.Graph(id='mainGraph', figure=px.line(),
                                    style={'flex': '1'}),
                      ])
                  ])

# Comments
commentModal = html.Div(
    id="commentModal",
    style={
        'display': 'none',
        'position': 'absolute',
        'top': '50%',
        'left': '50%',
                'transform': 'translate(-50%, -50%)',
                'background-color': 'white',
                'padding': '20px',
                'border-radius': '10px',
                'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.75)',
                'z-index': '9999'  # Ensures this div overlays other components
    },
    children=[
        dcc.Markdown("Comments"),
        html.Button("X", id='closeComments'),


        html.Div(children=[
            dcc.Input(id='commentInput', type='text', value='Comment'),
            dcc.Input(id='usernameInput', type='text', value='Name'),
            html.Button(id='addComment')])
    ]

)

# Alerts
alertTwoContainer = html.Div(children=[html.Div(id='alert2div',
                                                style={
                                                    'display': 'none',
                                                    'position': 'absolute',
                                                    'top': '50%',
                                                    'left': '50%',
                                                    'width': 400,
                                                    'height': 150,
                                                    'transform': 'translate(-50%, -50%)',
                                                    'background-color': 'white',
                                                    'padding': '20px',
                                                    'border-radius': '10px',
                                                    'box-shadow': '0px 0px 10px 0px rgba(0,0,0,0.75)',
                                                    'z-index': '9999',  # Ensures this div overlays other components
                                                    'flex-direction': 'column',
                                                    'align-items': 'center',
                                                    'justify-content': 'center'
                                                },
                                                children=[
                                                    dcc.Markdown('Warning: ', style={
                                                        'fontSize': 24, 'fontWeight': 'bold'}),
                                                    dcc.Markdown('Message', id='alert2', style={
                                                        'fontSize': 24}),
                                                    html.Button(
                                                        'Close', id='closeAlert2', style={})
                                                ]
                                                )])
# Alert 2
alertOneContainer = html.Div(children=[html.Div(id='alert1div', style={'display': 'none',  'backgroundColor': 'white', 'border': '5px solid black', 'margin': 10, 'align-items': 'center', 'width': 750, 'height': 50, 'flex-direction': 'row', },
                                                children=[

    dcc.Markdown('Click Data: ', style={
        'fontSize': 24, 'fontWeight': 'bold', 'padding': 10}),
    dcc.Markdown('Message', id='alert1', style={'fontSize': 24}),
    html.Button('Close', id='closeAlert1', style={'margin-left': 20})
]
)])

# Box 1
sensorHeader = dcc.Markdown('Sensors', style={
    'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
sensorDropdown = dcc.Checklist(options=data.columns,  style={
                               'fontSize': 20, 'margin': 10, 'display': 'none'}, inline=True, labelStyle={'width': '33%'})
sensorText = dcc.Markdown(
    'Select which sensors you wish to see in the graph above.', style={'textAlign': 'center', 'fontSize': 18})

xAxis_dropdown_3D = dcc.Dropdown(
    value=data.columns[0], options=data.columns, style={'width': '50%', 'display': 'block'})
yAxis_dropdown_3D = dcc.Dropdown(
    value=data.columns[1], options=data.columns, style={'width': '50%', 'display': 'block'})
zAxis_dropdown_3D = dcc.Dropdown(
    value=data.columns[2], options=data.columns, style={'width': '50%', 'display': 'block'})

xAxisText = dcc.Markdown(
    'x axis: ', style={'margin-left': 50, 'fontSize': 20, 'width': '50%', 'display': 'block'})
yAxisText = dcc.Markdown(
    'y axis: ', style={'margin-left': 50, 'fontSize': 20, 'width': '50%', 'display': 'block'})
zAxisText = dcc.Markdown(
    'z axis: ', style={'margin-left': 50, 'fontSize': 20, 'width': '50%', 'display': 'block'})



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

# Box 3

faultFinderHeader = html.H3("Fault Finder", style={'margin': '0'})

faultFinderText = dcc.Markdown(
    'Search for:', style={'fontSize': 20, 'width': '25%'})
faultFinder = dcc.Dropdown(value='Unlabelled Data Point', options=[
                           'Unlabelled Data Points', 'No Fault', 'Fault 1', 'Fault 2', 'Fault 3'], style={'width': '90%'})



# Box 4
AI_header = dcc.Markdown('Automatically Label Faults', style={
                         'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })
AI_text1 = dcc.Markdown(
    "Follow the steps to auto label the data set.  This will suggest the time and duration of faults. ", style={'textAlign': 'center', 'fontSize': 20})

AI_text2 = dcc.Markdown('Sensors: ', style={
    'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, })

AI_text3 = dcc.Markdown(
    "Start by selecting the sensors associated with the fault:", style={'margin-left': 10, 'fontSize': 20})

AI_text4 = dcc.Markdown('Feature Reduction: ', style={
    'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, })

AI_text5 = dcc.Markdown(
    "Clustering on many sensors can be poor.  Reduce the feature set through either PCA (recommended) or autoencoding.", style={'margin-left': 10, 'fontSize': 20})
AI_text6 = dcc.Markdown(
    'Reduction Method:', style={'margin-left': 10, 'width': '50%'})

AI_text7 = dcc.Markdown('Reduced Size:', id='reducedSizeMarkdown', style={
    'margin-left': 10, 'width': '50%', 'display': 'none'})

AI_text8 = dcc.Markdown('Clustering: ', style={
    'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, })

reducedSize = dcc.Input(type='number', id='reducedSize', style={
    'align-self': 'center', 'width': '100%', 'height': '90%', 'fontSize': 20, 'display': 'none'})

uploadNewAutoencoder = html.Button(
    children='Train autoencoder',
    style={
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'display': 'none'
    },
)

AI_text9 = dcc.Markdown(
    "Select the clustering algorithm.  Use K-means if you know the number of faults or a Neural Network if you have pre-labelled data.  Use DBSCAN if neither apply.", style={'margin-left': 10, 'fontSize': 20})


AI_text10 = dcc.Markdown('Clustering Method:', style={
                         'margin-left': 10, 'width': '50%'})

AI_text11 = dcc.Markdown('No. Clusters (K)', id='kMeansMarkdown',  style={
    'margin-left': 10, 'width': '50%'})
K = dcc.Input(type='number', value=3, style={
    'align-self': 'center', 'width': '100%', 'height': '90%', 'fontSize': 20})


AI_sensorChecklist = html.Div(style={'width': '100%', 'overflow': 'scroll'}, children=[
    dcc.Checklist(
        id='sensor-checklist', options=[], inline=True, labelStyle={'width': '25%', 'fontSize': 14})
])

AI_selectButtons = html.Div(style={'display': 'flex'}, children=[
    html.Button(
        "Select all", id='select-all', style=Styles.AI_button1),
    html.Button(
        "Deselect all", id='deselect-all', style=Styles.AI_button1),
    html.Button(
        "Select Sensors in Graph", id='graphSensors', style=Styles.AI_button1)
])

clusterMethod = dcc.Dropdown(
    options=['K Means', 'DBSCAN', 'Neural Network (Supervised)'], value='K Means', style={'width': '100%'})
reductionMethod = dcc.Dropdown(
    options=["PCA", "Auto-encoding", "None"], value='None', style={'width': '100%'})


# Box 5
stat1 = dcc.Markdown('No. Labels Placed: ', style={
                     'margin-left': 10, 'fontSize': 20})
stat2 = dcc.Markdown('No. Types Labels Placed: ', style={
                     'margin-left': 10, 'fontSize': 20})
stat3 = dcc.Markdown('Data points unlabelled: ', style={
                     'margin-left': 10, 'fontSize': 20})


# Box 6
uploadData = dcc.Upload(

    children=html.Div([

        html.A('Select Files to Upload'),
    ]),
    style={
        'width': '100%',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px',
        'display': 'block'
    },
    # Allow multiple files to be uploaded
    multiple=False
)

uploadTrainingData = dcc.Upload(

    children='Select data to train a new neural network',

    style={
        'display': 'none',
        'height': '60px',
        'lineHeight': '60px',
        'borderWidth': '1px',
        'borderStyle': 'dashed',
        'borderRadius': '5px',
        'textAlign': 'center',
        'margin': '10px'
    },

    multiple=True
)

AI_text12 = dcc.Markdown('Epsilon:',  style={
    'margin-left': 10, 'width': '50%', 'display': 'block'})

epsSlider = dcc.Slider(min=0, max=2,  marks={i: str(i) for i in range(0, 2)}, step=0.1, value=0.1)

AI_text13 = dcc.Markdown('Min Value:', style={
                         'margin-left': 10, 'width': '50%', 'display': 'none'})

minPtsSlider = dcc.Slider(min=1, max=60,  marks={i: str(
    i) for i in range(0, 60, 5)}, step=1, value=9)


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


navigatorTitle = dcc.Markdown(
    "Navigator", style={'margin': '20', 'fontSize': 24, 'fontWeight': 'bold', 'textAlign': 'center'})
navigationText = dcc.Markdown(
    'Search for:', style={'margin-left': 10, 'width': '25%'})
navigationButtons = html.Div(style={'flex-direction': 'row', 'justify-content': 'center', 'display': 'flex', 'align-items': 'center', 'width': '100%'}, children=[
    html.Button('Previous', id='findPrev', style={
                'width': 80, 'height': 25, 'margin': 10, 'fontSize': 16}),
    html.Button('Next', id='findNext', style={
                'width': 80, 'height': 25, 'margin': 10, 'fontSize': 16})
])


xAxis = html.Div(id='xAxisDropdownContainer', style=Styles.AxisDropdown, children=[
                 xAxisText, xAxis_dropdown_3D])
yAxis = html.Div(id='yAxisDropdownContainer', style=Styles.AxisDropdown, children=[
                 yAxisText, yAxis_dropdown_3D])
zAxis = html.Div(id='zAxisDropdownContainer', style=Styles.AxisDropdown, children=[
                 zAxisText, zAxis_dropdown_3D])

labelTitle = dcc.Markdown("Manually Label Faults", style={
                          'height': '20%', 'fontSize': 24, 'fontWeight': 'bold', 'textAlign': 'center'})

# Box 5
Box5text = dcc.Markdown('Statistics', style={
    'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })

Box6text = dcc.Markdown('Import or Export Data', style={
                        'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', })

downloadData = dcc.Download(id="downloadData")

dcc.Upload(children='Click to upload data', multiple=False, style={
    'width': '100%',
    'height': '60px',
    'lineHeight': '60px',
    'borderWidth': '1px',
    'borderStyle': 'dashed',
    'borderRadius': '5px',
    'textAlign': 'center',
    'margin': '10px'},)
