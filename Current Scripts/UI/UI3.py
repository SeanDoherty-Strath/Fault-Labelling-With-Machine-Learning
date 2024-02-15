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


# Internal Components and Functions
from Components import mainGraph, zoom, pan, fourGraphs, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, stats, faultFinder, alert, AI_checkbox
from Components import title, sensorDropdown, sensorHeader, labelDropdown, stat3, faultFinderHeader, faultFinderText, stat1, stat2, exportName, exportHeader, exportLocation, exportConfirm, AI_header, AI_text1, clusterMethod, AI_text2, reductionMethod, AI_input1, AI_input2, AI_input3, AI_input4
from myFunctions import changeText, updateGraph, performKMeans, performPCA, performDBSCAN, findBestParams

app = dash.Dash(__name__)

# DATA
data = pd.read_csv("Data/UpdatedData.csv")
data = data.drop(data.columns[[1, 2, 3]], axis=1)  # Remove extra columns
data = data.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column
data['labels'] = [-1]*data.shape[0]

# GLOBAL VARIABLES
shapes = []  # An array which stores rectangles, to visualise labels
currentPoint = 0  # The current point, for navigation
# -1 for non label, 0 for no fault, 1 for fault 1, 2 for fault 2 etc


# What proportion of the screen is shown
x_0 = 0
x_1 = 5000

colours = [['green'], ['red'], ['orange'], [
    'yellow'], ['pink'], ['purple'], ['lavender'], ['blue'], ['brown'], ['grey']]

# t is used to switch between time based view and 3D based view
t = None


# Define the layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, blue, #000000)', 'height': '100vh', 'display': 'flex', 'justify-content': 'center', 'flex-direction': 'column', 'align-items': 'center'}, children=[
    alert,
    html.Div(style={'height': '5%', 'width': '90%'},
             children=[title]),

    # Top Box
    html.Div(style={'overflow': 'scroll', 'width': '90%', 'height': '55%', 'margin': '20px', 'border-radius': '10px', 'padding': '20px', 'background-color': 'white'},
             children=[
        # dcc.Markdown(id='latestAction', children='Latest Action: '),
        html.Button('Switch View', id='switchView'),


        html.Div(style={'flex-direction': 'row', 'display': 'flex', 'width': '100%', 'height': '100%', }, children=[



            mainGraph,
            html.Div(id='ClusterColourContainer', style={
                     "display": "block", 'flex': 0.5, 'height': '100%'}, children=[
                         dcc.Dropdown(
                             style={'display': 'none'}, id=f'dropdown-0'),
                         dcc.Dropdown(
                             style={'display': 'none'}, id=f'dropdown-1'),
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
            ]
            ),

            html.Div(id='ClusterDropdownContainer', style={"display": "block", 'flex': 0.5, 'height': '100%'}, children=[
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
            html.Div(style={'overflow': 'scroll', 'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white'},
                     children=[
                # zoom,
                # pan
                dcc.Markdown('Labeller', style={
                             'fontSize': 26, 'fontWeight': 'bold', 'textAlign': 'center', }),

                html.Button(children='Start Label', id='labelButton', style={
                    'width': '100%', 'height': 40, 'fontSize': 16}),
                labelDropdown,
                html.Button('Remove Label', id='removeLabels', style={
                    'width': '50%'}),
                html.Button('Undo Label', id='undoLabel', style={
                    'width': '50%'}),
                html.Button('Change Label', id='changeLabel', style={
                    'width': '50%'}),
                html.Button('Move Label', id='moveLabel', style={
                    'width': '50%'}),
            ]),

            html.Div(
                style={'border-radius': '10px', 'width': '100%', 'height': '6%'}),

            #  Box 3
            html.Div(style={'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white'},
                     children=[
                html.Div(style={'display': 'flex', 'flex-direction': 'column', 'justify-content': 'center', 'text-align': 'left', 'align-items': 'center'},
                         children=[
                    faultFinderHeader,
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
            dcc.Markdown('Algorithms: ', style={
                'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, }),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown(
                    'Clustering Method:', style={'margin-left': 10, 'width': '50%'}),
                clusterMethod,
            ]),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown(
                    'Reduction Method:', style={'margin-left': 10, 'width': '50%'}),
                reductionMethod
            ]),


            dcc.Markdown('Parameters: ', style={
                'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, }),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown('No. Clusters (K)', id='kMeansMarkdown',  style={
                             'margin-left': 10, 'width': '50%'}),
                dcc.Input(type='number', id='K', style={
                    'align-self': 'center', 'width': '100%', 'height': '90%', 'fontSize': 20})
            ]),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown('Reduced Size:', id='reducedSizeMarkdown', style={
                             'margin-left': 10, 'width': '50%'}),
                dcc.Input(type='number', id='reducedSize', style={
                    'align-self': 'center', 'width': '100%', 'height': '90%', 'fontSize': 20})
            ]),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown('Epsilon:', id='epsilon', style={
                             'margin-left': 10, 'width': '50%'}),
                html.Div(style={'width': '100%'}, children=[
                    dcc.Slider(id='eps-slider', min=1, max=50,  marks={i: str(i) for i in range(1, 50, 5)}, step=1, value=32)]),
            ]),
            html.Div(style={'display': 'flex', }, children=[
                dcc.Markdown('Min Value:', id='minVal', style={
                             'margin-left': 10, 'width': '50%'}),
                html.Div(style={'width': '100%'}, children=[
                    dcc.Slider(id='minVal-slider', min=1, max=50,  marks={i: str(i) for i in range(1, 20, 2)}, step=1, value=9)]),
            ]),


            dcc.Markdown('Sensors: ', style={
                'fontSize': 22, 'fontWeight': 'bold', 'margin-left': 10, }),
            html.Div(style={'display': 'flex', }, children=[

                html.Button(
                    "Select all", id='select-all'),
                html.Button(
                    "Deselect all", id='deselect-all'),
            ]),
            html.Div(style={'width': '100%', 'height': 150, 'overflow': 'scroll'}, children=[
                dcc.Checklist(
                    id='sensor-checklist', options=data.columns[1:53], inline=True, labelStyle={'width': '33%'})
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
                dcc.Markdown(
                    'Shape clicked', id='shape-clicked'),
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


                ]
                )
            ])
        ]),
    ]),
])


# This function limits sensors to 4 at a time
@app.callback(
    Output(sensorDropdown, 'value'),
    Input(sensorDropdown, 'value')
)
def updatedText(values):
    if (values == None):
        return ['xmeas_1']
    if (len(values) > 4):
        values = values[1:]

    return values

# This function exports data to a downloadable csv


@app.callback(
    Output('downloadData', 'data'),
    Input(exportConfirm, 'n_clicks'),
    State(exportName, 'value')
)
def exportCSV(clicked, fileName):
    if clicked is None:
        raise PreventUpdate

    csv_filename = fileName + '.csv'
    data.to_csv(csv_filename, index=False)
    return dcc.send_file(csv_filename)


# This (behemoth) function stores the majority of user features.
# This is unavoidable, because all features which use 'mainGraph' must be in the same function
# Proceed with caution...

@app.callback(

    Output(mainGraph, 'figure'),
    Output('labelButton', 'children'),

    Output('removeLabels', 'n_clicks'),
    Output('undoLabel', 'n_clicks'),
    Output('findPrev', 'n_clicks'),
    Output('findNext', 'n_clicks'),

    Output(stat1, 'children'),
    Output(stat2, 'children'),
    Output(stat3, 'children'),
    Output(alert, 'is_open'),
    Output(alert, 'children'),

    Output(sensorDropdown, 'style'),
    Output('startAutoLabel', 'n_clicks'),
    Output('clusterDropdown', 'options'),
    Output('clusterDropdown', 'value'),
    Output('ClusterDropdownContainer', 'style'),


    Input(sensorDropdown, 'value'),
    State(labelDropdown, 'value'),
    Input('switchView', 'n_clicks'),
    Input('labelButton', 'n_clicks'),

    Input('removeLabels', 'n_clicks'),
    Input('undoLabel', 'n_clicks'),
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

    State('sensor-checklist', 'value'),
    State(clusterMethod, 'value'),
    State(reductionMethod, 'value'),

    State(mainGraph, 'relayoutData'),
    State('K', 'value'),
    State('reducedSize', 'value'),
    State('eps-slider', 'value'),
    State('minVal-slider', 'value')

)
def updateGraph(sensorDropdown, labelDropdown, switchViewButtonClicks, labelButtonClicks, removeLabelClick, undoLabelClick, findPrevClicked, findNextClicked, faultFinder, clickData, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, newAutoLabel, clusterDropdownOptions, clusterDropdownValue, sensorChecklist, clusterMethod, reductionMethod, relayoutData, K, reducedSize, eps, minVal):

    global shapes
    global colours
    global x_0
    global x_1

    if (newAutoLabel == None):
        newAutoLabel = 0
    labelButtonTitle = 'New Label'

    alert = False
    alertMessage = ''

    if (switchViewButtonClicks == None):
        switchViewButtonClicks = 0

    if (findNextClicked == None):
        findNextClicked = 0

    if (labelButtonClicks == None):
        labelButtonClicks = 0

    if relayoutData and 'xaxis.range[0]' in relayoutData.keys():
        x_0 = relayoutData.get('xaxis.range[0]')
        x_1 = relayoutData.get('xaxis.range[1]')

    if (switchViewButtonClicks % 3 == 0):
        ClusterDropdownContainer = {"display": "none"}
        maximum = 1

        # add data
        selectData = []
        for i in range(len(sensorDropdown)):
            name = sensorDropdown[i]
            yaxis = 'y' + str(i+1)

            if (data.loc[:, sensorDropdown[i]].max() > maximum):
                maximum = data.loc[:, sensorDropdown[i]].max()

            selectData.append(go.Scatter(
                y=data.loc[:, sensorDropdown[i]], name=name, yaxis=yaxis, opacity=1-0.2*i))

        if (labelButtonClicks % 2 == 0):

            dragMode = 'pan'
            # New label pressed

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
                if (x1 > 20000):
                    x1 = 20000

                shapesToDelete = []
                if shapes != []:
                    for shape in range(len(shapes)):

                        # if a previous label is inside this one, delete it.  This saves compuation cost
                        if x0 < shapes[shape]['x0'] and x1 > shapes[shape]['x1']:

                            shapesToDelete.append(shape)

                for shape in range(len(shapesToDelete)-1, -1, -1):

                    del shapes[shapesToDelete[shape]]

                shapes.append({
                    'type': 'rect',
                    'x0': x0,
                    'x1': x1,
                    'y0': 0,
                    'y1': 0.05,
                    'fillcolor': colours[labelDropdown][0],
                    'yref': 'paper',
                    'name': labelDropdown
                },)

                index_to_delete = next((index for index, shape in enumerate(
                    shapes) if shape['name'] == 'highlight'), None)
                if index_to_delete is not None:
                    if (shapes[index_to_delete]['x0'] < x1 and shapes[index_to_delete]['x1'] > x0):
                        del shapes[index_to_delete]

                data['labels'][x0:x1] = [labelDropdown] * (x1-x0)

                clusterDropdownOptions = list(set(data['labels']))
                clusterDropdownValue = clusterDropdownOptions

        elif (labelButtonClicks % 2 == 1):
            dragMode = 'select'
            labelButtonTitle = "Confirm Label"

        if (removeLabelClick == 1 and shapes != []):

            shapes.clear()
            data['labels'][0:len(data['labels'])] = [-1] * \
                (len(data['labels']))

        if (undoLabelClick == 1 and shapes != []):
            shapes.pop()
            shapes.pop()

        global currentPoint
        if (faultFinder == 'Unlabelled Data Point'):
            target = -1
        elif (faultFinder == 'No Fault'):
            target = 0
        elif (faultFinder == 'Fault 1'):
            target = 1
        elif (faultFinder == 'Fault 2'):
            target = 2
        elif (faultFinder == 'Fault 3'):
            target = 3

        if (findNextClicked == 1):

            index_to_delete = next((index for index, shape in enumerate(
                shapes) if shape['name'] == 'highlight'), None)
            if index_to_delete is not None:
                del shapes[index_to_delete]

            if (int(currentPoint) == len(data['labels'])):
                # Create an alert to informt that there are no furher ponts
                alert = True
                alertMessage = 'You have reached the end of the data'
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
                    alert = True
                    alertMessage = "No label exists"
                    x_0 = 0
                    x_1 = 20000
                else:
                    x_0 = start - round((end-start)*0.2)
                    x_1 = end + round((end-start)*0.2)

                shapes.append({
                    'type': 'rect',
                    'x0': start,
                    'x1': end,
                    'y0': 0,
                    'y1': 1,
                    'fillcolor': 'grey',
                    'opacity': 0.5,
                    'yref': 'paper',
                    'name': 'highlight',
                    'editable': False
                },)

        if (findPrevClicked == 1):
            # if (target == 'Unlabelled Data Point'):
            # Find the index of the shape to be deleted (by name)
            index_to_delete = next((index for index, shape in enumerate(
                shapes) if shape['name'] == 'highlight'), None)
            if index_to_delete is not None:
                del shapes[index_to_delete]

            if (int(currentPoint) == 0):
                # Create an alert to informt that there are no furher ponts
                alert = True
                alertMessage = 'You have reached the start of the data'
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
                    alert = True
                    alertMessage = "No label exists"
                    x_0 = 0
                    x_1 = 20000
                else:
                    x_0 = start - round((end-start)*0.2)
                    x_1 = end + round((end-start)*0.2)

                shapes.append({
                    'type': 'rect',
                    'x0': start,
                    'x1': end,
                    'y0': 0,
                    'y1': 1,
                    'fillcolor': 'grey',
                    'opacity': 0.5,
                    'yref': 'paper',
                    'name': 'highlight',
                    'editable': False
                },)

        if (newAutoLabel == 1):

            if (sensorChecklist == []):
                alert = True
                alertMessage = 'Select sensors for auto-detection.'
            else:

                shapes = []
                df = data.loc[:, sensorChecklist]

                # change this to PCA
                if (reductionMethod == 'PCA'):

                    if (reducedSize == None or reducedSize < 2):
                        alert = True
                        alertMessage = 'Wrong value input for PCA. Data reduction has failed.'

                    else:
                        df = performPCA(df, reducedSize)

                if (clusterMethod == 'K Means'):
                    if (K == None or K < 0):
                        alert = True
                        alertMessage = 'Wrong value input for K Means. Clustering has failed.'
                    else:
                        if (K > 10 or K <= 1):
                            alert = True
                            alertMessage = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = performKMeans(df, K)

                elif (clusterMethod == 'DBSCAN'):
                    # left in for wrong input
                    if (False):
                        alert = True
                        alertMessage = 'Wrong value input for K Means. Clustering has failed.'
                    else:
                        n = len(sensorChecklist)
                        data['labels'] = performDBSCAN(df, n)

                shapes = []
                x0 = 0
                x1 = x0

                clusterDropdownOptions = list(set(data['labels']))
                clusterDropdownValue = clusterDropdownOptions

                if (len(clusterDropdownOptions) > 10):
                    alert = True
                    alertMessage = "Auto Labelling produced too many clusters.  Try increasing epsilon or decreaseing minimum value."
                else:
                    # labels = [0, 0, 0, 1, 1, 1, 2, 3, 4]
                    for i in range(1, len(data['labels'])):

                        if data['labels'][i] != data['labels'][i-1]:

                            x1 = i

                            shapes.append({
                                'type': 'rect',
                                'x0': x0,
                                'x1': x1,
                                'y0': 0,
                                'y1': 0.05,
                                # 'fillcolor': colours[labels[x0]][0],
                                'fillcolor': 'white',
                                'yref': 'paper',
                                'name': 'area'+str(data['labels'][x0])
                            },)

                            x0 = i

                    shapes.append({
                        'type': 'rect',
                                'x0': x0,
                                'x1': len(data['labels']),
                                'y0': 0,
                                'y1': 0.05,
                                # 'fillcolor': colours[labels[x0]][0],
                                'fillcolor': 'white',
                                'yref': 'paper',
                                'name': 'area'+str(data['labels'][x0])
                    },)
                    x_0 = 0
                    x_1 = 20000

        print('Dragmode: ', dragMode)
        layout = go.Layout(legend={'x': 0, 'y': 1.2}, xaxis=dict(range=[x_0, x_1]), dragmode=dragMode, yaxis=dict(fixedrange=True, title='Sensor Value', color='blue'), yaxis2=dict(
            fixedrange=True, overlaying='y', color='orange', side='right'), yaxis3=dict(fixedrange=True, overlaying='y', color='green', side='left', position=0.001,), yaxis4=dict(fixedrange=True, overlaying='y', color='red', side='right'), shapes=shapes)

        if t is not None:
            selectData.append(
                go.Line(x=[t, t], y=[0, maximum], name='Selected Point', line=dict(color='black')))

        fig = {'data': selectData, 'layout': layout, }

        sensorDropdownStyle = {'display': 'block'}

    if (switchViewButtonClicks % 3 == 1):

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
                    if data['xmeas_1'][label] > x0 and data['xmeas_1'][label] < x1 and data['xmeas_2'][label] > y0 and data['xmeas_2'][label] < y1:
                        data.loc[label, 'labels'] = labelDropdown
                        print('identified: ', label)

        else:
            print('got here buddy')
            labelButtonTitle = 'Confirm Label'
            dragMode = 'select'

        ClusterDropdownContainer = {"display": "none"}
        # fig = px.scatter_3d(data, x='xmeas_1', y='xmeas_2', z='xmeas_3', text='Unnamed: 0', color_discrete_sequence=['black'])

        selectData = [go.Scatter(
            y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
                                                         xAxis_dropdown_3D], mode='markers', marker={'color': [colours[val][0] for val in data['labels']], })]
        layout = go.Layout(dragmode=dragMode)
        fig = {'data': selectData, 'layout': layout}

        # selectData = [go.Scatter(
        #     y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
        #                                                  xAxis_dropdown_3D], mode='markers',]

        # if t is not None:
        #     fig.add_scatter(x=[data[xAxis_dropdown_3D][t]], y=[
        #                     data[yAxis_dropdown_3D][t]],  marker=dict(color='black'))

    if (switchViewButtonClicks % 3 == 2):
        #  3D SCATTER PLOT

        ClusterDropdownContainer = {
            "display": "block", 'position': 'absolute', 'right': '30%'}

        selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:,
                                                                                zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers',
                                   marker={
            'size': 10,
            'opacity': 1,
            # 'color': 'red'
            # Set color based on 'labels' column
            'color': [colours[val][0] for val in data['labels']],
        },)]

        layout = go.Layout()

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

        sensorDropdownStyle = {'display': 'none'}

    labels = data['labels'].values.tolist()
    n = labels.count(-1)
    stat1 = 'Data points unlabelled: ' + str(n)
    n = len(set(labels))
    stat2 = 'No. Types Labels Placed: ', int(len(set(labels))-1)
    stat3 = 'No. Labels Placed: ', len(shapes)

    return fig, labelButtonTitle, 0, 0, 0, 0, stat1, stat2, stat3, alert, alertMessage, sensorDropdownStyle, 0, clusterDropdownOptions, clusterDropdownValue, ClusterDropdownContainer


@app.callback(
    Output('ClusterColourContainer', 'children'),
    Input('startAutoLabel', 'n_clicks'),
)
def autoLabelOptions(startAutoLabelClicks):
    dropdowns = [dcc.Markdown('Cluster Colouring:')]
    for i in range(len(set(data['labels']))):

        dropdowns.append(
            dcc.Markdown('Area ' + str(i+1))
        )
        dropdowns.append(
            dcc.Dropdown(
                style={'display': 'block'},
                id=f'dropdown-{i}',
                options=[
                    {'label': 'No Fault (green)', 'value': 0},
                    {'label': 'Fault 1 (Red)', 'value': 1},
                    {'label': 'Fault 2 (orange)', 'value': 2},
                    {'label': 'Fault 2 (yellow)', 'value': 3}]
            )
        )
    for i in range(len(set(data['labels'])), 11):
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

    return dropdowns


@app.callback(

    Output(mainGraph, 'figure', allow_duplicate=True),
    Input('dropdown-0', 'value'),
    Input('dropdown-1', 'value'),
    Input('dropdown-2', 'value'),
    Input('dropdown-3', 'value'),
    Input('dropdown-4', 'value'),
    Input('dropdown-5', 'value'),
    Input('dropdown-6', 'value'),
    Input('dropdown-7', 'value'),
    Input('dropdown-8', 'value'),
    Input('dropdown-9', 'value'),
    State(mainGraph, 'figure'),
    prevent_initial_call=True,

)
def colorLabels(area0, area1, area2, area3, area4, area5, area6, area7, area8, area9, figure):

    colors = [area0, area1, area2, area3, area4,
              area5, area6, area7, area8, area9]
    for i in range(len(colors)):

        if colors[i] == None or colors[i] == '':
            colors[i] = -1

    # for shape in shapes:
    #     print(shape)
    #     for j in range(10):
    #         if shape.name == 'area'+str(j):
    for color in colors:
        print('Color:')
        print(color)
        for shape in shapes:
            if shape['name'] == 'area'+str(color):
                shape['fillcolor'] = colours[color][0]

    # x0 = 0
    # for i in range(1, len(data['labels'])):

    #     if data['labels'][i] != data['labels'][i-1]:

    #         x1 = i

    #         shapes.append({
    #             'type': 'rect',
    #             'x0': x0,
    #             'x1': x1,
    #             'y0': 0,
    #             'y1': 0.05,
    #             # 'fillcolor': colours[labels[x0]][0],
    #             # 'fillcolor': colors[data['labels'][x0]],
    #             'fillcolor': [colours[val][0] for val in data['labels']],
    #             'yref': 'paper',
    #             'name': 'misc'})

    #         x0 = i

    layout = go.Layout(legend={'x': 0, 'y': 1.2}, xaxis=dict(range=[0, 20000]), dragmode='pan', yaxis=dict(fixedrange=True, title='Sensor Value', color='blue'), yaxis2=dict(
        fixedrange=True, overlaying='y', color='orange', side='right'), yaxis3=dict(fixedrange=True, overlaying='y', color='green', side='left', position=0.001,), yaxis4=dict(fixedrange=True, overlaying='y', color='red', side='right'), shapes=shapes)

    figure = {'data': figure['data'], 'layout': layout}

    return figure


@app.callback(
    Output('points-output', 'children'),
    [Input(mainGraph, 'clickData')],
    State('switchView', 'n_clicks')


)
def display_coordinates(click_data, switchViewClicks):
    if (switchViewClicks == None):
        switchViewClicks = 0

    global t

    if (switchViewClicks % 2 == 0):
        if click_data is not None and 'points' in click_data:
            point = click_data['points'][0]

            t = point['x']

            return f'Time = {t}'

        else:
            return 'Click on a point to display its coordinates',
    elif (switchViewClicks % 2 == 1):
        if click_data is not None and 'points' in click_data:
            point = click_data['points'][0]
            t = point['pointNumber']
            return f'Time = {t}'

        else:
            return 'Click on a point to display its coordinates',


@app.callback(
    Output('shape-clicked', 'children'),
    [Input(mainGraph, 'clickData')],
    State('switchView', 'n_clicks')
)
def update_textbox(click_data, switchViewClicks):
    if (switchViewClicks == None):
        switchViewClicks = 0

    if click_data is None:
        return "Click on a shape to see its information"
    else:

        if (switchViewClicks % 2 == 0):
            # Find the shape that was clicked
            clicked_shape_info = None
            for shape in shapes:
                if (
                    shape['x0'] <= click_data['points'][0]['x'] and
                    shape['x1'] >= click_data['points'][0]['x']
                ):
                    clicked_shape_info = shape['name']

                    break

            if clicked_shape_info:
                return f"Clicked Shape Info: {clicked_shape_info}"
            else:
                return "No shape clicked"
        elif (switchViewClicks % 2 == 1):
            return '3D Plot'


@app.callback(
    Output('sensor-checklist', 'value'),
    Output('select-all', 'n_clicks'),
    Output('deselect-all', 'n_clicks'),
    Input('select-all', 'n_clicks'),
    Input('deselect-all', 'n_clicks')
)
def selectDeselectAll(selectClicks, deselectClicks):
    if selectClicks == None:
        selectClicks = 0
    if deselectClicks == None:
        deselectClicks = 0
    if selectClicks == 1:
        return data.columns[1:], 0, 0
    else:
        return [], 0, 0


@app.callback(
    Output('K', 'style'),
    Output('reducedSize', 'style'),
    Output('reducedSizeMarkdown', 'style'),
    Output('kMeansMarkdown', 'style'),
    Output('ClusterColourContainer', 'style'),

    Input(clusterMethod, 'value'),
    Input(reductionMethod, 'value'),
    Input('switchView', 'n_clicks')
)
def autoLabelStyles(clusterMethod, reductionMethod, switchView):

    K_style = {'display': 'none'}
    kMeansMarkdown = {'display': 'none'}
    reducedStyle_style = {'display': 'none'}
    reducedSizeMarkdown = {'display': 'none'}
    ClusterColourContainer = {'display': 'none'}

    if switchView == None:
        switchView = 0

    if (switchView % 2 == 0):

        ClusterColourContainer = {
            "display": "block", 'flex': 0.2, 'height': '100%'}
    else:
        ClusterColourContainer = {'display': 'none'}

    if (clusterMethod == 'K Means'):
        K_style = {'display': 'block', 'align-self': 'center',
                   'width': '100%', 'height': '90%', 'fontSize': 20}
        kMeansMarkdown = {'display': 'block',
                          'margin-left': 10, 'width': '50%'}

    if (reductionMethod == 'PCA'):
        reducedStyle_style = {'display': 'block', 'align-self': 'center',
                              'width': '100%', 'height': '90%', 'fontSize': 20}
        reducedSizeMarkdown = {'display': 'block',
                               'margin-left': 10, 'width': '50%'}

    return K_style, reducedStyle_style, reducedSizeMarkdown, kMeansMarkdown, ClusterColourContainer


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
