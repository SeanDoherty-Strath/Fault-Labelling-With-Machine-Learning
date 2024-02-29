# FAULT DETECTION AND LABELLING TOOL
# Sean Doherty, 202013008
# 4th Year Project

# External Libraries
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import pandas as pdv
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash.exceptions import PreventUpdate
import io
import base64  # Import base64 module

# Internal Components and Functions
from Components import mainGraph, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, faultFinder, alert
from Components import title, sensorDropdown, sensorHeader, labelDropdown, stat3, faultFinderHeader, faultFinderText, stat1, stat2, exportName, exportConfirm, AI_header, clusterMethod, reductionMethod
from myFunctions import performKMeans, performPCA, performDBSCAN, performAutoEncoding, findBestParams, knee_point

app = dash.Dash(__name__)

# DATA
data = pd.read_csv("Data/UpdatedData.csv")
data = data.drop(data.columns[[1, 2, 3]], axis=1)  # Remove extra columns
data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column
# 0 for non label, -1 for no fault, 2 for fault 1, 2 for fault 3 etc
data['labels'] = [0]*data.shape[0]
data['clusterLabels'] = [0]*data.shape[0]


# GLOBAL VARIABLES
shapes = []  # An array which stores rectangles, to visualise labels in the time domain
currentPoint = 0  # The current point, for navigation


# What proportion of the screen is shown
x_0 = 0
x_1 = 5000

colours = [['grey'], ['green'], ['red'], ['orange'], ['yellow'], ['pink'],
           ['purple'], ['lavender'], ['blue'], ['brown'], ['cyan']]

greyColours = [['#000000'], ['#E0E0E0'], ['#606060'], ['#404040'], [
    '#A0A0A0'], ['#FFFFFF'], ['#202020'], ['#808080'], ['#C0C0C0'],]


# t is used to switch between time based view and 3D based view
t = None

# Define the layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, blue, #000000)', 'height': '100vh', 'display': 'flex', 'justify-content': 'center', 'flex-direction': 'column', 'align-items': 'center'}, children=[

    html.Div(
        id='alert2div',
        style={
            'display': 'none',

            'backgroundColor': 'white',
            'border': '5px solid black',
            'border-radius': 15,
            'margin': 20,
            'align-items': 'center',

            'width': '90%',
            'height': 50,
            'flex-direction': 'row',
        },
        children=[
            dcc.Markdown('Warning: ', style={
                         'fontSize': 24, 'fontWeight': 'bold', 'padding': 10}),
            dcc.Markdown('Message', id='alert2', style={'fontSize': 24})
        ]
    ),

    html.Div(
        id='alert1div',
        style={
            'display': 'none',

            'backgroundColor': 'white',
            'border': '5px solid black',
            'border-radius': 15,
            'margin': 20,
            'align-items': 'center',

            'width': '90%',
            'height': 50,
            'flex-direction': 'row',
        },
        children=[
            dcc.Markdown('Click Data: ', style={
                         'fontSize': 24, 'fontWeight': 'bold', 'padding': 10}),
            dcc.Markdown('Message', id='alert1', style={'fontSize': 24})
        ]
    ),
    # html.Div(style={'height': '5%', 'width': '90%'},
    #          children=[title,
    #                    ]),

    # Top Box
    html.Div(style={'overflow': 'scroll', 'width': '90%', 'height': '55%', 'margin': '20px', 'border-radius': '10px', 'padding': '20px', 'background-color': 'white'},
             children=[



        # dcc.Markdown(id='latestAction', children='Latest Action: '),
        html.Button('Switch View', id='switchView'),


        html.Div(style={'flex-direction': 'row', 'display': 'flex', 'width': '100%', 'height': '100%', }, children=[

            html.Div(id='ClusterColourContainer', style={
                     "display": "none", 'flex': '1'}, children=[
                         dcc.Dropdown(
                             style={'display': 'none', }, id=f'dropdown-0'),
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
                             style={'display': 'none'}, id='colorNow'),

            ]
            ),
            mainGraph,
            # alert,


            html.Div(id='ClusterDropdownContainer', style={"display": "none"}, children=[
                dcc.Markdown("Show Clusters", style={
                             'fontSize': 30, 'fontWeight': 'bold'}),
                dcc.Checklist(id='clusterDropdown', options=[],
                              style={}),
            ])
        ])

        # dcc.Graph(id='tempGraph')
        # zoom
        # fourGraphs
    ]),

    # Bottom boxes
    html.Div(style={'width': '90%', 'height': '35%',  'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin': '20px'},
             children=[
        # Box 1
        html.Div(style={'overflow': 'scroll', 'border-radius': '10px', 'width': '24%', 'height': '100%',  'background-color': 'white'},
                 children=[
            sensorHeader,
            sensorDropdown,
            xAxis_dropdown_3D,
            yAxis_dropdown_3D,
            zAxis_dropdown_3D]),


        html.Div(style={'width': '24%', 'height': '100%', }, children=[
            #  Box 2
            html.Div(style={'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white', },
                     children=[
                         html.Div(style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'left',  'align-items': 'center'},
                                  children=[
                             dcc.Markdown("Fault Labeller", style={'margin': '20', 'fontSize': 24, 'fontWeight': 'bold'})]),
                labelDropdown,
                html.Button(children='Start Label', id='labelButton', style={
                    'width': '100%', 'height': 40, 'fontSize': 16}),

                html.Button('Remove Labels', id='removeLabels', style={'height': 40,
                                                                       'width': '100%'}),

            ]),

            html.Div(
                style={'border-radius': '10px', 'width': '100%', 'height': '6%'}),

            #  Box 3
            html.Div(style={'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white'},
                     children=[
                html.Div(style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'left',  'align-items': 'center'},
                         children=[
                    dcc.Markdown("Navigator", style={
                                 'margin': '20', 'fontSize': 24, 'fontWeight': 'bold'}),
                    html.Div(style={'flex-direction': 'column', 'display': 'flex', 'width': '90%'}, children=[
                        faultFinderText,
                        faultFinder
                    ]),
                    html.Div(style={'flex-direction': 'row'}, children=[
                        html.Button('Prev', id='findPrev'),
                        html.Button('Next', id='findNext'),
                    ])


                    # dcc.Markdown('0', id='currentPoint')
                ])
            ]),]),

        # Box 4
        html.Div(style={'overflow': 'scroll',  'border-radius': '10px', 'width': '24%', 'height': '100%', 'background-color': 'white'},
                 children=[
            AI_header,

            dcc.Markdown('Sensors: ', style={
                'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, }),
            html.Div(style={'display': 'flex', }, children=[

                html.Button(
                    "Select all", id='select-all'),
                html.Button(
                    "Deselect all", id='deselect-all'),
                html.Button(
                    "Select Sensors in Graph", id='graphSensors'),
            ]),
            html.Div(style={'width': '100%', 'height': 150, 'overflow': 'scroll'}, children=[
                dcc.Checklist(
                    id='sensor-checklist', options=data.columns[1:53], inline=True, labelStyle={'width': '33%'})
            ]),

            dcc.Markdown('Feature Reduction: ', style={
                'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, }),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown(
                    'Reduction Method:', style={'margin-left': 10, 'width': '50%'}),
                reductionMethod
            ]),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown('Reduced Size:', id='reducedSizeMarkdown', style={
                             'margin-left': 10, 'width': '50%'}),
                dcc.Input(type='number', id='reducedSize', style={
                    'align-self': 'center', 'width': '100%', 'height': '90%', 'fontSize': 20})
            ]),


            dcc.Markdown('Clustering: ', style={
                'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, }),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown(
                    'Clustering Method:', style={'margin-left': 10, 'width': '50%'}),
                clusterMethod,
            ]),



            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown('No. Clusters (K)', id='kMeansMarkdown',  style={
                             'margin-left': 10, 'width': '50%'}),
                dcc.Input(type='number', id='K', value=3, style={
                    'align-self': 'center', 'width': '100%', 'height': '90%', 'fontSize': 20})
            ]),

            html.Div(id='epsilon', style={'display': 'flex'}, children=[
                dcc.Markdown('Epsilon:',  style={
                             'margin-left': 10, 'width': '50%'}),
                html.Div(style={'width': '100%'}, children=[
                    dcc.Slider(id='eps-slider', min=0, max=2,  marks={i: str(i) for i in range(0, 2)}, step=0.1, value=0.1)]),
            ]),
            html.Div(id='minVal', style={'display': 'flex', }, children=[
                dcc.Markdown('Min Value:', style={
                             'margin-left': 10, 'width': '50%'}),
                html.Div(style={'width': '100%'}, children=[
                    dcc.Slider(id='minVal-slider', min=1, max=60,  marks={i: str(i) for i in range(0, 60, 5)}, step=1, value=9)]),
            ]),





            html.Button(children='Start Now', id='startAutoLabel', style={
                'width': '100%', 'fontSize': 20})


        ]),

        # Box 5
        html.Div(style={'width': '24%', 'height': '100%', 'overflow': 'scroll'},
                 children=[
            html.Div(style={'overflow': 'scroll', 'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white'},
                     children=[
                         dcc.Markdown('Stats', style={
                             'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', }),
                stat1, stat2, stat3,
                # dcc.Markdown(
                #     'Shape clicked', id='shape-clicked'),
                dcc.Markdown('Points Output:', id='points-output', style={"display": "block"}),]
            ),
            html.Div(
                style={'border-radius': '10px', 'width': '100%', 'height': '6%', }),

            # Box6
            html.Div(style={'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white', 'display': 'flex', 'justify-content': 'center', },
                     children=[
                html.Div(style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'align-items': 'center', 'width': '100%', },
                         children=[
                    dcc.Markdown('Export File to CSV', style={
                         'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', }),
                    html.Div(style={'display': 'flex', 'width': '100%', }, children=[
                        # dcc.Markdown('File Name:', style={
                        # }),
                        exportName,
                        exportConfirm
                    ]),

                    dcc.Download(
                        id="downloadData"),
                    dcc.Upload(
                        id='upload-data',
                        children=html.Div([
                            'Drag and Drop or ',
                            html.A('Select Files')
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

                    dcc.Markdown(id='upload-data-text', children='')


                ]
                )
            ])
        ]),
    ]),
])


# This function limits sensors to 4 at a time
@app.callback(
    Output(sensorDropdown, 'value', allow_duplicate=True),
    Input(sensorDropdown, 'value'),
    prevent_initial_call=True
)
def updatedText(values):

    if (values == None):

        return [data.columns[0]]
    if (len(values) > 4):
        values = values[1:]

    return values

# This function exports data to a downloadable csv


@app.callback(
    Output('downloadData', 'data'),
    Input(exportConfirm, 'n_clicks'),
    State(exportName, 'value')
)
def exportCSV(exportClicked, fileName):
    if exportClicked is None:
        raise PreventUpdate

    csv_filename = fileName + '.csv'
    data.to_csv(csv_filename, index=False)
    return dcc.send_file(csv_filename)


@app.callback(Output('upload-data-text', 'children'),
              Output(sensorDropdown, 'options', allow_duplicate=True),
              Output(sensorDropdown, 'value'),
              Output('sensor-checklist', 'options', allow_duplicate=True),
              Output(mainGraph, 'figure', allow_duplicate=True),
              Output(xAxis_dropdown_3D, 'value'),
              Output(xAxis_dropdown_3D, 'options'),
              Output(yAxis_dropdown_3D, 'value'),
              Output(yAxis_dropdown_3D, 'options'),
              Output(zAxis_dropdown_3D, 'value'),
              Output(zAxis_dropdown_3D, 'options'),
              [Input('upload-data', 'contents')],
              prevent_initial_call=True,)
def update_output(contents):
    global data
    global shapes
    global x_0
    global x_1

    if contents is not None:
        print('contents', contents)
        content_type, content_string = contents.split(',')
        decoded = io.StringIO(base64.b64decode(content_string).decode('utf-8'))
        data = pd.read_csv(decoded)
        data['labels'] = data['labels'] = [0]*data.shape[0]
        data['clusterLabels'] = [0]*data.shape[0]
        sensors = data.columns[1:len(data.columns)-2]

        x_0 = 0
        x_1 = data.shape[0]

        # shapes = []
        layout = go.Layout(xaxis=dict(range=[x_0, x_1]))

        return 'Success', sensors, [data.columns[1]], sensors, {'layout': layout}, data.columns[1], data.columns, data.columns[2], data.columns, data.columns[3], data.columns
    else:
        raise PreventUpdate
    # return html.Div([
    #     html.H5('Uploaded File Content:'),
    #     dcc.Textarea(
    #         value=df.to_string(),
    #         readOnly=True,
    #         style={'width': '100%', 'height': '300px'}
    #     ),
    #     html.Hr(),
    #     html.H5('Dataframe Info:'),
    #     html.Pre(df.info())
    # ])


# This (behemoth) function stores the majority of user features.
# This is unavoidable, because all features which use 'mainGraph' must be in the same function
# Proceed with caution...

@app.callback(

    Output(mainGraph, 'figure'),
    Output('labelButton', 'children'),

    Output('removeLabels', 'n_clicks'),
    # Output('undoLabel', 'n_clicks'),
    Output('findPrev', 'n_clicks'),
    Output('findNext', 'n_clicks'),

    Output(stat1, 'children'),
    Output(stat2, 'children'),
    Output(stat3, 'children'),
    # Output(alert, 'is_open'),
    # Output(alert, 'children'),
    Output('alert2div', 'style', allow_duplicate=True),
    Output('alert2', 'children', allow_duplicate=True),

    Output(sensorDropdown, 'style'),
    Output('startAutoLabel', 'n_clicks'),
    Output('clusterDropdown', 'options'),
    Output('clusterDropdown', 'value'),
    Output('ClusterDropdownContainer', 'style'),
    Output('ClusterColourContainer', 'style'),
    Output(xAxis_dropdown_3D, 'style'),
    Output(yAxis_dropdown_3D, 'style'),
    Output(zAxis_dropdown_3D, 'style'),
    Input(sensorDropdown, 'value'),
    State(labelDropdown, 'value'),
    Input('switchView', 'n_clicks'),
    Input('labelButton', 'n_clicks'),

    Input('removeLabels', 'n_clicks'),
    # Input('undoLabel', 'n_clicks'),
    Input('findPrev', 'n_clicks'),
    Input('findNext', 'n_clicks'),
    Input(faultFinder, 'value'),
    Input(mainGraph, 'clickData'),
    # Input('currentPoint', 'children'),
    Input(xAxis_dropdown_3D, 'value'),
    Input(yAxis_dropdown_3D, 'value'),
    Input(zAxis_dropdown_3D, 'value'),
    Input('startAutoLabel', 'n_clicks'),
    Input('clusterDropdown', 'options'),
    Input('clusterDropdown', 'value'),

    Input('colorNow', 'n_clicks'),

    State('sensor-checklist', 'value'),
    State(clusterMethod, 'value'),
    State(reductionMethod, 'value'),

    State(mainGraph, 'relayoutData'),
    State('K', 'value'),
    State('reducedSize', 'value'),
    State('eps-slider', 'value'),
    State('minVal-slider', 'value'),
    State('alert2div', 'style'),
    State('alert2', 'children'),
    prevent_initial_call=True,

)
def updateGraph(sensorDropdown, labelDropdown, switchViewButtonClicks, labelButtonClicks, removeLabelClick, findPrevClicked, findNextClicked, faultFinder, clickData, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, newAutoLabel, clusterDropdownOptions, clusterDropdownValue, colorNow, sensorChecklist, clusterMethod, reductionMethod, relayoutData, K, reducedSize, eps, minVal, alert2div, alert2):

    global shapes
    global colours
    global x_0
    global x_1
    global currentPoint

    # Set Buttons to 0
    if (newAutoLabel == None):
        newAutoLabel = 0
    if (switchViewButtonClicks == None):
        switchViewButtonClicks = 0

    if (findNextClicked == None):
        findNextClicked = 0

    if (labelButtonClicks == None):
        labelButtonClicks = 0

    if (removeLabelClick == None):
        removeLabelClick = 0

    # Set default output values
    labelButtonTitle = 'New Label'

    ClusterColourContainer = {"display": "none"}

    ClusterDropdownContainer = {"display": "none"}
    xAxis_dropdown_3D_style = {"display": "none"}
    yAxis_dropdown_3D_style = {"display": "none"}
    zAxis_dropdown_3D_style = {"display": "none"}
    alert2div['display'] = 'none'

    if (removeLabelClick == 1):
        data.loc[:, 'labels'] = 0

    # Take note of initial x0 and x1 values
    if relayoutData and 'xaxis.range[0]' in relayoutData.keys():
        x_0 = relayoutData.get('xaxis.range[0]')
        x_1 = relayoutData.get('xaxis.range[1]')

    if (switchViewButtonClicks % 3 == 0):

        # add data
        selectData = []
        maximum = 1
        # Used to find the maximum value of all the sensor values

        for i in range(len(sensorDropdown)):

            name = sensorDropdown[i]
            yaxis = 'y' + str(i+1)

            if (data.loc[:, sensorDropdown[i]].max() > maximum):
                maximum = data.loc[:, sensorDropdown[i]].max()

            selectData.append(go.Scatter(
                y=data.loc[:, sensorDropdown[i]], name=name, yaxis=yaxis, opacity=1-0.2*i))

        if (labelButtonClicks % 2 == 0):

            dragMode = 'pan'

            if relayoutData is not None and 'selections' in relayoutData.keys():

                x0 = relayoutData['selections'][0]['x0']
                x1 = relayoutData['selections'][0]['x1']
                x0 = round(x0)
                x1 = round(x1)

                if (x0 > x1):
                    temp = x0
                    x0 = x1
                    x1 = temp

                if (x0 < 0):
                    x0 = 0
                if (x1 > data.shape[0]):
                    x1 = data.shape[0]

                data['labels'][x0:x1] = [labelDropdown] * (x1-x0)

                # clusterDropdownOptions = list(set(data['labels']))
                # clusterDropdownValue = clusterDropdownOptions

                # shapesToDelete = []
                # if shapes != []:
                #     for shape in range(len(shapes)):

                #         # if a previous label is inside this one, delete it.  This saves compuation cost
                #         if x0 < shapes[shape]['x0'] and x1 > shapes[shape]['x1']:

                #             shapesToDelete.append(shape)

                # for shape in range(len(shapesToDelete)-1, -1, -1):

                #     del shapes[shapesToDelete[shape]]

                # shapes.append({
                #     'type': 'rect',
                #     'x0': x0,
                #     'x1': x1,
                #     'y0': 0,
                #     'y1': 0.05,
                #     'fillcolor': colours[labelDropdown][0],
                #     'yref': 'paper',
                #     'name': labelDropdown
                # },)

                # index_to_delete = next((index for index, shape in enumerate(
                #     shapes) if shape['name'] == 'highlight'), None)
                # if index_to_delete is not None:
                #     if (shapes[index_to_delete]['x0'] < x1 and shapes[index_to_delete]['x1'] > x0):
                #         del shapes[index_to_delete]

        elif (labelButtonClicks % 2 == 1):

            dragMode = 'select'
            labelButtonTitle = "Confirm Label"

        if (findNextClicked == 1):

            if (faultFinder == 'Unlabelled Data Point'):
                target = 0
            elif (faultFinder == 'No Fault'):
                target = 1
            elif (faultFinder == 'Fault 1'):
                target = 2
            elif (faultFinder == 'Fault 2'):
                target = 3
            elif (faultFinder == 'Fault 3'):
                target = 4

            # index_to_delete = next((index for index, shape in enumerate(
            #     shapes) if shape['name'] == 'highlight'), None)
            # if index_to_delete is not None:
            #     del shapes[index_to_delete]

            if (int(currentPoint) == len(data['labels'])):
                # Create an alert to informt that there are no furher ponts

                alert2div['display'] = 'flex'
                alert2 = 'You have reached the end of the data.'
            else:
                start = -1
                end = -1
                for i in range(int(currentPoint), len(data['labels'])):
                    if (data['labels'][i] == target):
                        start = i
                        for j in range(i, len(data['labels'])):
                            if (data['labels'][j] != target):
                                end = j
                                currentPoint = str(end)
                                break
                        if (end == -1):
                            end = len(data['labels'])
                            currentPoint = str(end)
                        break
                if (start == -1):
                    # There is no exisiting label

                    alert2div['display'] = 'flex'
                    alert2 = 'No label exists.'
                    x_0 = 0
                    x_1 = data.shape[0]
                else:
                    x_0 = start - round((end-start)*0.2)
                    x_1 = end + round((end-start)*0.2)

                # shapes.append({
                #     'type': 'rect',
                #     'x0': start,
                #     'x1': end,
                #     'y0': 0,
                #     'y1': 1,
                #     'fillcolor': 'grey',
                #     'opacity': 0.5,
                #     'yref': 'paper',
                #     'name': 'highlight',
                #     'editable': False
                # },)

        if (findPrevClicked == 1):
            # if (target == 'Unlabelled Data Point'):
            # Find the index of the shape to be deleted (by name)
            # index_to_delete = next((index for index, shape in enumerate(
            #     shapes) if shape['name'] == 'highlight'), None)
            # if index_to_delete is not None:
            #     del shapes[index_to_delete]

            if (int(currentPoint) == 0):
                # Create an alert to informt that there are no furher ponts
                alert2div['display'] = 'flex'
                alert2 = 'You have reached the start of the data'

            else:
                start = -1
                end = -1

                for i in range(int(currentPoint)-1, 0, -1):
                    if (data['labels'][i] == target):
                        end = i
                        start = -1
                        for j in range(i, 0, -1):
                            if (data['labels'][j] != target):
                                start = j
                                currentPoint = str(start)
                                break
                        if (start == -1):
                            start = 0
                            currentPoint = str(start)
                        break
                if (end == -1):
                    # There is no exisiting label

                    alert2div['display'] = 'flex'
                    alert2 = 'No label exists'
                    x_0 = 0
                    x_1 = data.shape[0]
                else:
                    x_0 = start - round((end-start)*0.2)
                    x_1 = end + round((end-start)*0.2)

                # shapes.append({
                #     'type': 'rect',
                #     'x0': start,
                #     'x1': end,
                #     'y0': 0,
                #     'y1': 1,
                #     'fillcolor': 'grey',
                #     'opacity': 0.5,
                #     'yref': 'paper',
                #     'name': 'highlight',
                #     'editable': False
                # },)

        if (newAutoLabel == 1):

            if (sensorChecklist == []):

                alert2div['display'] = 'flex'
                alert2 = 'Select sensors for auto-detection.'

            else:

                df = data.loc[:, sensorChecklist]

                if (reductionMethod == 'PCA'):

                    if (reducedSize == None or reducedSize < 2):

                        alert2div['display'] = 'flex'
                        alert2 = 'Wrong value input for PCA. Data reduction has failed.'

                    else:

                        df = performPCA(df, reducedSize)

                elif (reductionMethod == 'Auto-encoding'):

                    df = performAutoEncoding(df)

                if (clusterMethod == 'K Means'):
                    if (K == None or K < 0):

                        alert2div['display'] = 'flex'
                        alert2 = 'Wrong value input for K Means. Clustering has failed.'

                    else:
                        if (K > 10 or K <= 1):

                            alert2div['display'] = 'flex'
                            alert2 = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performKMeans(df, K)

                elif (clusterMethod == 'DBSCAN'):
                    # left in for wrong input
                    if (eps == None or minVal == None):

                        alert2div['display'] = 'flex'
                        alert2 = 'Incorrect parameter for eps or min points.'

                    else:
                        # data['labels'] = [0]*data.shape[0]
                        # n = len(sensorChecklist)
                        # data['clusterLabels'] = performDBSCAN(df, n)
                        n = len(sensorChecklist)
                        temp = performDBSCAN(df, eps, minVal)
                        if len(list(set(temp))) >= 10:
                            alert2div['display'] = 'flex'
                            alert2 = 'DBSCAN produced too many clusters. Try increasing epsilon or decreasing min points.'
                        elif (len(list(set(temp))) == 1):
                            alert2div['display'] = 'flex'
                            alert2 = 'DBSCAN produced only outliers. Try decreasing epsilon or decreasing min points.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performDBSCAN(
                                df, eps, minVal)

                # clusterDropdownOptions = list(set(data['clusterLabels']))
                # clusterDropdownValue = clusterDropdownOptions

                ClusterColourContainer = {
                    'display': 'block', 'width': 200, 'padding': 20}

                x_0 = 0
                x_1 = data.shape[0]

        # Go through labels and shown all the shapes
        shapes = []
        x0 = 0
        x1 = x0

        # if data['clusterLabels'].all() == data['labels'].all():
        #     data['clusterLabels'] = [0]*data.shape[0]

        for i in range(1, len(data['labels'])):

            if data['labels'][i] != data['labels'][i-1]:

                x1 = i

                shapes.append({
                    'type': 'rect',
                    'x0': x0,
                    'x1': x1,
                    'y0': 0,
                    'y1': 0.05,
                    'fillcolor': colours[data['labels'][x0]][0],
                    'yref': 'paper',
                    'opacity': 1,
                    'name': str(data['labels'][x0])
                },)

                x0 = i

        shapes.append({
            'type': 'rect',
            'x0': x0,
            'x1': len(data['labels']),
            'y0': 0,
            'y1': 0.05,
            'fillcolor': colours[data['labels'][x0]][0],
            'yref': 'paper',
            'opacity': 1,
            'name': str(data['labels'][x0])
        },)

        if len(set(data['clusterLabels'])) != 1:
            for i in range(1, len(data['clusterLabels'])):

                if data['clusterLabels'][i] != data['clusterLabels'][i-1]:

                    x1 = i

                    shapes.append({
                        'type': 'rect',
                        'x0': x0,
                        'x1': x1,
                        'y0': 0,
                        'y1': 0.05,
                        'fillcolor': greyColours[data['clusterLabels'][x0]][0],
                        'yref': 'paper',
                        'opacity': 0.2,
                        'name': 'area'+str(data['clusterLabels'][x0])
                    },)

                    x0 = i

            shapes.append({
                'type': 'rect',
                'x0': x0,
                'x1': len(data['labels']),
                'y0': 0,
                'y1': 0.05,
                'fillcolor': greyColours[data['clusterLabels'][x0]][0],
                'yref': 'paper',
                'opacity': 0.2,
                'name': 'area'+str(data['clusterLabels'][x0])
            },)

        layout = go.Layout(legend={'x': 0, 'y': 1.2}, xaxis=dict(range=[x_0, x_1]),  dragmode=dragMode, yaxis=dict(fixedrange=True, title='Sensor Value', color='blue'), yaxis2=dict(
            fixedrange=True, overlaying='y', color='orange', side='right'), yaxis3=dict(fixedrange=True, overlaying='y', color='green', side='left', position=0.001,), yaxis4=dict(fixedrange=True, overlaying='y', color='red', side='right'), shapes=shapes)

        if t is not None:
            selectData.append(
                go.Line(x=[t, t], y=[0, maximum], name='Selected Point', line=dict(color='black')))

        fig = {'data': selectData, 'layout': layout, }

        sensorDropdownStyle = {'display': 'block'}

    if (switchViewButtonClicks % 3 == 1):
        xAxis_dropdown_3D_style = {"display": "block"}
        yAxis_dropdown_3D_style = {"display": "block"}

        sensorDropdownStyle = {'display': 'none'}

        if (labelButtonClicks % 2 == 0):

            dragMode = 'zoom'

            if 'selections' in relayoutData.keys():

                x0 = relayoutData['selections'][0]['x0']
                x1 = relayoutData['selections'][0]['x1']
                y0 = relayoutData['selections'][0]['y0']
                y1 = relayoutData['selections'][0]['y1']

                if (x0 > x1):
                    temp = x0
                    x0 = x1
                    x1 = temp
                if (y0 > y1):
                    temp = y0
                    y0 = y1
                    y1 = temp

                for label in range(len(data['labels'])):
                    if data[xAxis_dropdown_3D][label] > x0 and data[xAxis_dropdown_3D][label] < x1 and data[yAxis_dropdown_3D][label] > y0 and data[yAxis_dropdown_3D][label] < y1:
                        data.loc[label, 'labels'] = labelDropdown

        else:

            labelButtonTitle = 'Confirm Label'
            dragMode = 'select'

        if (newAutoLabel == 1):

            if (sensorChecklist == []):

                alert2div['display'] = 'flex'
                alert2 = 'Select sensors for auto-detection.'

            else:

                df = data.loc[:, sensorChecklist]

                if (reductionMethod == 'PCA'):

                    if (reducedSize == None or reducedSize < 2):

                        alert2div['display'] = 'flex'
                        alert2 = 'Wrong value input for PCA. Data reduction has failed.'

                    else:

                        df = performPCA(df, reducedSize)

                elif (reductionMethod == 'Auto-encoding'):

                    df = performAutoEncoding(df)

                if (clusterMethod == 'K Means'):
                    if (K == None or K < 0):

                        alert2div['display'] = 'flex'
                        alert2 = 'Wrong value input for K Means. Clustering has failed.'

                    else:
                        if (K > 10 or K <= 1):

                            alert2div['display'] = 'flex'
                            alert2 = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performKMeans(df, K)

                elif (clusterMethod == 'DBSCAN'):
                    # left in for wrong input
                    if (eps == None or minVal == None):

                        alert2div['display'] = 'flex'
                        alert2 = 'Incorrect parameter for eps or min points.'

                    else:
                        # data['labels'] = [0]*data.shape[0]
                        # n = len(sensorChecklist)
                        # data['clusterLabels'] = performDBSCAN(df, n)
                        n = len(sensorChecklist)
                        temp = performDBSCAN(df, eps, minVal)
                        if len(list(set(temp))) >= 10:
                            alert2div['display'] = 'flex'
                            alert2 = 'DBSCAN produced too many clusters. Try increasing epsilon or decreasing min points.'
                        elif (len(list(set(temp))) == 1):
                            alert2div['display'] = 'flex'
                            alert2 = 'DBSCAN produced only outliers. Try decreasing epsilon or decreasing min points.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performDBSCAN(
                                df, eps, minVal)

                # clusterDropdownOptions = list(set(data['clusterLabels']))
                # clusterDropdownValue = clusterDropdownOptions

                ClusterColourContainer = {
                    'display': 'block', 'width': 200, 'padding': 20}

                x_0 = 0
                x_1 = data.shape[0]

                selectData = [go.Scatter(
                    y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
                                                                 xAxis_dropdown_3D], text=data.loc[:, 'clusterLabels'], mode='markers', marker={'color': [greyColours[val][0] for val in data['clusterLabels']], })]
        else:
            selectData = [go.Scatter(
                y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
                                                             xAxis_dropdown_3D], text=data.loc[:, 'labels'], mode='markers', marker={'color': [colours[val][0] for val in data['labels']], })]
        layout = go.Layout(dragmode=dragMode, yaxis=dict(
            title=yAxis_dropdown_3D), xaxis=dict(
            title=xAxis_dropdown_3D))
        fig = {'data': selectData, 'layout': layout}

        # selectData = [go.Scatter(
        #     y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
        #                                                  xAxis_dropdown_3D], mode='markers',]

        # if t is not None:
        #     fig.add_scatter(x=[data[xAxis_dropdown_3D][t]], y=[
        #                     data[yAxis_dropdown_3D][t]],  marker=dict(color='black'))

    if (switchViewButtonClicks % 3 == 2):
        #  3D SCATTER PLOT
        xAxis_dropdown_3D_style = {"display": "block"}
        yAxis_dropdown_3D_style = {"display": "block"}
        zAxis_dropdown_3D_style = {"display": "block"}

        ClusterDropdownContainer = {
            "display": "block", 'flex': 0.5}

        if (newAutoLabel == 1):

            if (sensorChecklist == []):

                alert2div['display'] = 'flex'
                alert2 = 'Select sensors for auto-detection.'

            else:

                df = data.loc[:, sensorChecklist]

                if (reductionMethod == 'PCA'):

                    if (reducedSize == None or reducedSize < 2):

                        alert2div['display'] = 'flex'
                        alert2 = 'Wrong value input for PCA. Data reduction has failed.'

                    else:

                        df = performPCA(df, reducedSize)

                elif (reductionMethod == 'Auto-encoding'):

                    df = performAutoEncoding(df)

                if (clusterMethod == 'K Means'):
                    if (K == None or K < 0):

                        alert2div['display'] = 'flex'
                        alert2 = 'Wrong value input for K Means. Clustering has failed.'

                    else:
                        if (K > 10 or K <= 1):

                            alert2div['display'] = 'flex'
                            alert2 = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performKMeans(df, K)

                elif (clusterMethod == 'DBSCAN'):
                    # left in for wrong input
                    if (eps == None or minVal == None):

                        alert2div['display'] = 'flex'
                        alert2 = 'Incorrect parameter for eps or min points.'

                    else:
                        # data['labels'] = [0]*data.shape[0]
                        # n = len(sensorChecklist)
                        # data['clusterLabels'] = performDBSCAN(df, n)
                        n = len(sensorChecklist)
                        temp = performDBSCAN(df, eps, minVal)
                        if len(list(set(temp))) >= 10:
                            alert2div['display'] = 'flex'
                            alert2 = 'DBSCAN produced too many clusters. Try increasing epsilon or decreasing min points.'
                        elif (len(list(set(temp))) == 1):
                            alert2div['display'] = 'flex'
                            alert2 = 'DBSCAN produced only outliers. Try decreasing epsilon or decreasing min points.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performDBSCAN(
                                df, eps, minVal)

                # clusterDropdownOptions = list(set(data['clusterLabels']))
                # clusterDropdownValue = clusterDropdownOptions

                x_0 = 0
                x_1 = data.shape[0]

                # clusterDropdownOptions = list(set(data['clusterLabels']))
                # clusterDropdownValue = clusterDropdownOptions
                ClusterColourContainer = {
                    'display': 'block', 'width': 200, 'padding': 20}

                selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:,
                                                                                        zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers',
                                           marker={
                    'size': 10,
                    'opacity': 1,
                    # 'color': 'red'
                    # Set color based on 'labels' column
                    'color': [greyColours[val][0] for val in data['clusterLabels']],
                    # 'color': colours[0][0],
                },)]

                # if (len(clusterDropdownOptions) > 10):
                #     alert = True
                #     alertMessage = "Auto Labelling produced too many clusters.  Try increasing epsilon or decreaseing minimum value."
                # else:
                #     # labels = [0, 0, 0, 1, 1, 1, 2, 3, 4]
                #     for i in range(1, len(data['labels'])):

                #         if data['clusterLabels'][i] != data['clusterLabels'][i-1]:

                #             x1 = i

                #             shapes.append({
                #                 'type': 'rect',
                #                 'x0': x0,
                #                 'x1': x1,
                #                 'y0': 0,
                #                 'y1': 0.05,
                #                 # 'fillcolor': colours[labels[x0]][0],
                #                 'fillcolor': colours[0][0],
                #                 'yref': 'paper',
                #                 'name': 'area'+str(data['clusterLabels'][x0])
                #             },)

                #             x0 = i

                # shapes.append({
                #     'type': 'rect',
                #             'x0': x0,
                #             'x1': len(data['labels']),
                #             'y0': 0,
                #             'y1': 0.05,
                #             # 'fillcolor': colours[labels[x0]][0],
                #             'fillcolor': colours[0][0],
                #             'yref': 'paper',
                #             'name': 'area'+str(data['clusterLabels'][x0])
                # },)

        # selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:,
        #                                                                         zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers',
        #                            marker={
        #     'size': 10,
        #     'opacity': 1,
        #     # 'color': 'red'
        #     # Set color based on 'labels' column
        #     # 'color': [colours[val][0] for val in data['labels']],
        #     'color': colours[0][0],
        # },)]
        else:
            selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:,
                                                                                    zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers',
                                       marker={
                'size': 10,
                'opacity': 1,
                # 'color': 'red'
                # Set color based on 'labels' column
                'color': [colours[val][0] for val in data['labels']],
                # 'color': da,
            },)]
        sensorDropdownStyle = {'display': 'none'}
        layout = go.Layout(xaxis=dict(
            title=xAxis_dropdown_3D), yaxis=dict(
            title=yAxis_dropdown_3D), )

        fig = {'data': selectData, 'layout': layout}

        # cluster_dataframes = []

        # grouped = data.groupby('labels')

        # Creating separate dataframes for each group
        # cluster_dataframes = {}
        # for label, group in grouped:
        #     cluster_dataframes[label] = group

        # fig = px.scatter_3d()
        # if t is not None:
        #     fig.add_scatter3d(x=[data[xAxis_dropdown_3D][t]], y=[data[yAxis_dropdown_3D][t]], z=[
        #                       data[zAxis_dropdown_3D][t]], marker=dict(color='black'))

        # for cluster in cluster_dataframes:
        #     if clusterDropdownValue:
        #         if cluster in clusterDropdownValue:
        #             fig.add_trace(px.scatter_3d(cluster_dataframes[cluster], x=xAxis_dropdown_3D, y=yAxis_dropdown_3D, opacity=0.5, z=zAxis_dropdown_3D,
        #                                         color_discrete_sequence=colours[cluster]).data[0])
        #     else:
        #         alert = True
        #         alertMessage = 'No Clusters Placed'

    labels = data['labels'].values.tolist()
    n = labels.count(0)
    stat1 = 'Unlabelle Data: ' + str(n)
    stat2 = 'Labelled Data: ' + str(len(data['labels']) - n)
    n = len(set(labels))
    stat3 = 'No. Labels Placed: ', int(len(set(labels)))

    return fig, labelButtonTitle, 0, 0, 0, stat1, stat2, stat3, alert2div,  alert2, sensorDropdownStyle, 0, clusterDropdownOptions, clusterDropdownValue, ClusterDropdownContainer, ClusterColourContainer, xAxis_dropdown_3D_style, yAxis_dropdown_3D_style, zAxis_dropdown_3D_style


@app.callback(
    Output('ClusterColourContainer', 'children'),
    Input('startAutoLabel', 'n_clicks'),
)
def autoLabelOptions(startAutoLabelClicks):

    # if startAutoLabelClicks == None:
    #     startAutoLabelClicks = 0
    # if startAutoLabelClicks > 0:
    dropdowns = [dcc.Markdown('Cluster Colouring:')]
    for i in range(len(set(data['clusterLabels']))):

        dropdowns.append(
            dcc.Markdown('Area ' + str(i+1))
        )
        dropdowns.append(
            dcc.Dropdown(
                style={'display': 'block', 'width': 200},
                id=f'dropdown-{i}',
                options=[
                    {'label': 'No Fault (Green)', 'value': 1},
                    {'label': 'Fault 1 (Red)', 'value': 2},
                    {'label': 'Fault 2 (Orange)', 'value': 3},
                    {'label': 'Fault 3 (Yellow)', 'value': 4},
                    {'label': 'Fault 4 (Pink)', 'value': 5},
                    {'label': 'Fault 5 (Purple)', 'value': 6},
                    {'label': 'Fault 6 (Lavender)', 'value': 7},
                    {'label': 'Fault 7 (Blue)', 'value': 8},
                    {'label': 'Fault 8 (Brown)', 'value': 9},
                    {'label': 'Fault 9 (Cyan)', 'value': 10}
                ]
            )
        )

    for i in range(len(set(data['clusterLabels'])), 11):
        dropdowns.append(
            dcc.Markdown('Area ' + str(i+1), style={'display': 'none'},)
        )
        dropdowns.append(
            dcc.Dropdown(
                style={'display': 'none'},
                id=f'dropdown-{i}',

                options=[]
            )
        )
    dropdowns.append(
        html.Button('Confirm Labels', id='colorNow')
    )

    return dropdowns
    # else:
    #     raise PreventUpdate


@app.callback(
    # Output(mainGraph, 'figure', allow_duplicate=True),
    Output('colorNow', 'n_clicks'),
    Output('ClusterColourContainer', 'style', allow_duplicate=True),
    Output('alert2div', 'style', allow_duplicate=True),
    Output('alert2', 'children', allow_duplicate=True),
    Input('colorNow', 'n_clicks'),
    State('dropdown-0', 'value'),
    State('dropdown-1', 'value'),
    State('dropdown-2', 'value'),
    State('dropdown-3', 'value'),
    State('dropdown-4', 'value'),
    State('dropdown-5', 'value'),
    State('dropdown-6', 'value'),
    State('dropdown-7', 'value'),
    State('dropdown-8', 'value'),
    State('dropdown-9', 'value'),
    # Input('dropdown-0', 'value'),
    # Input('dropdown-1', 'value'),
    # Input('dropdown-2', 'value'),
    # Input('dropdown-3', 'value'),
    # Input('dropdown-4', 'value'),
    # Input('dropdown-5', 'value'),
    # Input('dropdown-6', 'value'),
    # Input('dropdown-7', 'value'),
    # Input('dropdown-8', 'value'),
    # Input('dropdown-9', 'value'),
    State(mainGraph, 'figure'),
    State(mainGraph, 'relayoutData'),
    State('switchView', 'n_clicks'),
    State('alert2div', 'style'),
    prevent_initial_call=True,

)
def colorLabels(colorNow, area0, area1, area2, area3, area4, area5, area6, area7, area8, area9, figure, relayoutData, switchView, alert2div):

    if colorNow == None or colorNow == 0:
        raise PreventUpdate
    else:

        print(alert2div['display'])
        global x_0
        global x_1

        alert2div['display'] = 'none'

        alertMessage = ''

        if (switchView is None):
            switchView = 0

        # if (colorNow is None):
        #     colorNow = 0

        areas = [area0, area1, area2, area3, area4,
                 area5, area6, area7, area8, area9]

        ClusterColourContainer = {'display': 'none'}

        for i in range(len(set(data['clusterLabels']))):
            if areas[i] == None:

                alert2div['display'] = 'flex'
                alertMessage = 'Not all dropdowns were full. Labelling may be wrong.'

        for i in range(len(data['labels'])):
            for j in range(len(areas)):
                if j == data['clusterLabels'][i]:
                    if areas[j] == None:
                        data.loc[i, 'labels'] = 0
                    else:
                        data.loc[i, 'labels'] = areas[j]

        data['clusterLabels'] = [0]*data.shape[0]
        ClusterColourContainer = {'display': 'none'}

        return 0, ClusterColourContainer, alert2div, alertMessage

    # if (switchView % 3 == 0):
    #     for shape in shapes:
    #         for label in range(len(labels)):
    #             if shape['name'] == 'area'+str(label) and labels[label] is not None:
    #                 shape['fillcolor'] = colours[labels[label]][0]
    #                 data['labels'] = [0]*data.shape[0]
    #                 # shape['name'] = labels[label]
    #                 data['labels'][shape['x0']:shape['x1']] = [
    #                     labels[label]] * (shape['x1']-shape['x0'])

    #     if relayoutData and 'xaxis.range[0]' in relayoutData.keys():
    #         x_0 = relayoutData.get('xaxis.range[0]')
    #         x_1 = relayoutData.get('xaxis.range[1]')

    #     if 'shapes' in figure['layout']:
    #         figure['layout']['shapes'] = shapes

    #     return figure

    # elif (switchView % 3 == 2):

    #     for i in range(len(data['clusterLabels'])):

    #         for label in range(len(labels)):

    #             if data['clusterLabels'][i] == label:

    #                 if labels[label] is not None:
    #                     data['labels'] = labels[label]
    #             # if data['clusterLabels'][i] == labels[label]:
    #             #     data['labels'] = labels[label]

    #     selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:,
    #                                                                             zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers',
    #                                marker={
    #         'size': 10,
    #         'opacity': 1,
    #         # 'color': 'red'
    #         # Set color based on 'labels' column
    #         # 'color': [colours[val][0] for val in data['labels']],
    #         'color':  [colours[val][0] for val in data['labels']],
    #     },)]

    #     layout = go.Layout()

    #     fig = {'data': selectData, 'layout': layout}

    #     return fig


@app.callback(
    Output('points-output', 'children'),
    [Input(mainGraph, 'clickData')],
    State('switchView', 'n_clicks')

)
def display_coordinates(click_data, switchViewClicks):
    if (switchViewClicks == None):
        switchViewClicks = 0

    global t

    if (switchViewClicks % 3 == 0):
        if click_data is not None and 'points' in click_data:
            point = click_data['points'][0]

            t = point['x']

            return f'Time = {t}'

        else:
            return 'Click on a point to display its coordinates',
    elif (switchViewClicks % 3 == 2):
        if click_data is not None and 'points' in click_data:
            point = click_data['points'][0]
            t = point['pointNumber']
            return f'Time = {t}'

        else:
            return 'Click on a point to display its coordinates',


@app.callback(
    Output('alert1div', 'style'),
    Output('alert1', 'children'),
    [Input(mainGraph, 'clickData')],
    State('switchView', 'n_clicks'),
    State('alert1div', 'style'),
    State('alert1', 'children')
)
def update_textbox(click_data, switchViewClicks, alertstyle, alert):
    if (switchViewClicks == None):
        switchViewClicks = 0

    if click_data is None:
        raise PreventUpdate

    # if (len(set(data['clusterLabels'])) != 1):
    #     t = round(click_data['points'][0]['pointNumber'])
    #     label = data['clusterLabels'][t]
    #     alertstyle['display'] = 'flex'
    #     alert = 'This point (t = ' + str(t) + \
    #         ') belongs to Area ', str(label)

    #     return alertstyle, alert
    # else:
    alertstyle['style'] = 'none'
    labels = ['Unlabelled', 'No Fault', 'Fault 1', 'Fault 2', 'Fault 3',
              'Fault 4', 'Fault 5', 'Fault 6', 'Fault 7', 'Fault 8', 'Fault 9', 'Fault 10']

    t = round(click_data['points'][0]['pointNumber'])
    label = data['labels'][t]
    alertstyle['display'] = 'flex'
    alert = 'This point (t = ' + str(t) + \
        ') is labelled as ', str(labels[label])

    return alertstyle, alert

# def update_textbox(click_data, switchViewClicks):
#     if (switchViewClicks == None):
#         switchViewClicks = 0

#     print(click_data)

#     if click_data is None:
#         return "Click on a shape to see its information"
#     else:

#         if (switchViewClicks % 3 == 0):
#             # Find the shape that was clicked
#             clicked_shape_info = None
#             for shape in shapes:
#                 if (
#                     shape['x0'] <= click_data['points'][0]['x'] and
#                     shape['x1'] >= click_data['points'][0]['x']
#                 ):
#                     clicked_shape_info = shape['name']

#                     break

#             if clicked_shape_info:
#                 return f"Clicked Shape Info: {clicked_shape_info}"
#             else:
#                 return "No shape clicked"
#         elif (switchViewClicks % 3 == 1):
#             return '2D Plot'
#         elif (switchViewClicks % 3 == 2):
#             return '3D Plot'


@app.callback(
    Output('sensor-checklist', 'value', allow_duplicate=True),
    Output('select-all', 'n_clicks'),
    Output('deselect-all', 'n_clicks'),
    Output('graphSensors', 'n_clicks'),
    Input('select-all', 'n_clicks'),
    Input('deselect-all', 'n_clicks'),
    Input('graphSensors', 'n_clicks'),
    State(sensorDropdown, 'value'),
    State('switchView', 'n_clicks'),
    State(xAxis_dropdown_3D, 'value'),
    State(yAxis_dropdown_3D, 'value'),
    State(zAxis_dropdown_3D, 'value'),
    prevent_initial_call=True
)
def selectDeselectAll(selectClicks, deselectClicks, graphSensors, sensorDropdown, switchView, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D):
    if selectClicks == None:
        selectClicks = 0
    if deselectClicks == None:
        deselectClicks = 0
    if switchView == None:
        switchView = 0

    if selectClicks == 1:

        return data.columns[1:], 0, 0, 0
    elif deselectClicks == 1:
        return [], 0, 0, 0
    elif graphSensors == 1:
        if (switchView % 3 == 0):
            return sensorDropdown, 0, 0, 0
        elif (switchView % 3 == 1):
            return [xAxis_dropdown_3D, yAxis_dropdown_3D], 0, 0, 0
        elif (switchView % 3 == 2):
            return [xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D], 0, 0, 0
    else:
        raise PreventUpdate


@app.callback(
    Output('K', 'style'),
    Output('reducedSize', 'style'),
    Output('reducedSizeMarkdown', 'style'),
    Output('kMeansMarkdown', 'style'),
    Output('epsilon', 'style'),
    Output('minVal', 'style'),


    Input(clusterMethod, 'value'),
    Input(reductionMethod, 'value'),
    Input('switchView', 'n_clicks')
)
def autoLabelStyles(clusterMethod, reductionMethod, switchView):

    K_style = {'display': 'none'}
    kMeansMarkdown = {'display': 'none'}
    reducedStyle_style = {'display': 'none'}
    reducedSizeMarkdown = {'display': 'none'}
    epsStyle = {'display': 'none'}
    minValStyle = {'display': 'none'}

    if switchView == None:
        switchView = 0

    if (clusterMethod == 'K Means'):
        K_style = {'display': 'block', 'align-self': 'center',
                   'width': '100%', 'height': '90%', 'fontSize': 20}
        kMeansMarkdown = {'display': 'block',
                          'margin-left': 10, 'width': '50%'}

    if (clusterMethod == 'DBSCAN'):
        epsStyle = {'display': 'flex'}
        minValStyle = {'display': 'flex'}

    if (reductionMethod == 'PCA'):
        reducedStyle_style = {'display': 'block', 'align-self': 'center',
                              'width': '100%', 'height': '90%', 'fontSize': 20}
        reducedSizeMarkdown = {'display': 'block',
                               'margin-left': 10, 'width': '50%'}

    return K_style, reducedStyle_style, reducedSizeMarkdown, kMeansMarkdown, epsStyle, minValStyle


@app.callback(
    Output('minVal-slider', 'value'),
    Output('eps-slider', 'value'),
    Input(clusterMethod, 'value'),
    Input('sensor-checklist', 'value'),
    Input('reducedSize', 'value'),
    Input(reductionMethod, 'value')

)
def DBSCAN_parameterSelection(clusterMethod, sensorChecklist, reducedSize, reductionMethod):
    print(sensorChecklist)
    if clusterMethod == 'DBSCAN' and sensorChecklist != []:
        df = data.loc[:, sensorChecklist]
        if reductionMethod == 'PCA':
            if reducedSize != None:
                df = performPCA(df, reducedSize)
        elif reductionMethod == 'Auto-encoding':
            df = performAutoEncoding(df)

        n = len(df.columns)
        eps = knee_point(df, n + 1)

        return n+1, eps
    else:
        raise PreventUpdate


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
