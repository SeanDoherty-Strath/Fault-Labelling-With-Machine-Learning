# FAULT DETECTION AND LABELLING TOOL
# Sean Doherty, 202013008
# 4th Year Project

# IMPORTS
# External Libraries for UI
import dash
from dash import html, dcc, Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash.exceptions import PreventUpdate
import numpy as np

# External Libraries for Data handling
import pandas as pd
import tensorflow as tf

# Other Libraries
import time
import datetime
import io
import base64

# Internal Libraries
from InternalLibraries.ML_Functions import performKMeans, performPCA, performDBSCAN, findKneePoint, createAutoencoder, trainNeuralNetwork, useNeuralNetwork, testAccuracy
import InternalLibraries.Styles as Styles
import InternalLibraries.Components as Components
from InternalLibraries.Layout import mainLayout

# Global Variables
data = pd.DataFrame()
comments = pd.DataFrame()
classifierNeuralNetwork = tf.keras.models.load_model(
    "FaultLabeller/NeuralNetworks/multiclassNeuralNetwork")
autoencoderNeuralNetwork = tf.keras.models.load_model(
    "FaultLabeller/NeuralNetworks/autoencoder")
shapes = []  # An array which stores rectangles, to visualise labels in the time domain
navigationPoint = 0  # current navigation point
clickedPoint = None  # The current clicked point
x_0 = 0  # What proportion of the time graph is shown
x_1 = 5000

# START APP
app = dash.Dash(__name__)

# Define the layout
app.layout = mainLayout

# This function limits sensors to 4 at a time


@app.callback(
    Output(Components.sensorDropdown, 'value', allow_duplicate=True),
    Input(Components.sensorDropdown, 'value'),
    prevent_initial_call=True
)
def limitSensor(values):

    if (values == None):
        return [data.columns[0]]
    if (len(values) > 4):
        values = values[1:]

    return values

# This function exports data to a downloadable csv


@app.callback(
    Output('downloadData', 'data'),
    Input(Components.exportConfirm, 'n_clicks'),
    State(Components.exportName, 'value'),
    State(Components.includeCommentsButton, 'n_clicks'),
)
def exportCSV(exportClicked, fileName, includeCommentsButton):
    try:
        if exportClicked is None:
            raise PreventUpdate
        if data.empty:
            raise PreventUpdate

        if includeCommentsButton == None:
            includeCommentsButton = 0

        # Remove undesired data
        exportData = data.drop(columns=['clusterLabels'])
        exportData = exportData.rename(columns={'labels': 'primaryFault'})
        exportData = exportData.rename(columns={'secondary': 'secondaryFault'})
        exportData = exportData.rename(columns={'tertiary': 'tertiaryFault'})

        if includeCommentsButton % 2 == 0 and not comments.empty:
            exportData['commentMessage'] = comments.loc[:, 'commentMessage']
            exportData['commentTime'] = comments.loc[:, 'commentTime']
            exportData['commentUser'] = comments.loc[:, 'commentUser']

        exportData.to_csv(fileName+'.csv', index=False)
        return dcc.send_file(fileName+'.csv')
    except Exception as e:
        raise PreventUpdate

# This function updates the 'add comments' button


@app.callback(
    Output(Components.includeCommentsButton, 'children'),
    Input(Components.includeCommentsButton, 'n_clicks'),
)
def includeComments(nClicks):
    if nClicks is None:
        nClicks = 0

    if nClicks % 2 == 0:
        return 'Include Comments: Yes'
    if nClicks % 2 == 1:
        return 'Include Comments: No'

#  This function uploads data to the dashboard


@app.callback(Output(Components.title, 'children'),
              Output(Components.sensorDropdown,
                     'options', allow_duplicate=True),
              Output(Components.sensorDropdown, 'value'),
              Output('sensor-checklist', 'options', allow_duplicate=True),
              Output('mainGraph', 'figure', allow_duplicate=True),
              Output(Components.xAxis_dropdown_3D, 'value'),
              Output(Components.xAxis_dropdown_3D, 'options'),
              Output(Components.yAxis_dropdown_3D, 'value'),
              Output(Components.yAxis_dropdown_3D, 'options'),
              Output(Components.zAxis_dropdown_3D, 'value'),
              Output(Components.zAxis_dropdown_3D, 'options'),
              Input(Components.uploadData, 'contents'),
              Input(Components.uploadData, 'filename'),
              prevent_initial_call=True,)
def uploadData(contents, filename):
    try:
        global data
        global shapes
        global x_0
        global x_1
        global comments

        if contents is not None:
            #  split content
            contentType, content = contents.split(',')
            # decode data
            decoded = io.StringIO(base64.b64decode(content).decode('utf-8'))
            data = pd.read_csv(decoded)

            comments = pd.DataFrame()

            # Remove unwanted data
            if 'commentMessage' in data.columns and 'commentTime' in data.columns and 'commentUser' in data.columns:
                comments['commentTime'] = data.loc[:, 'commentTime']
                comments['commentUser'] = data.loc[:, 'commentUser']
                comments['commentMessage'] = data.loc[:, 'commentMessage']
                data.drop(columns=['commentMessage', 'commentUser',
                                   'commentTime'], inplace=True)

            if 'Unnamed: 0' in data.columns:
                data = data.rename(columns={'Unnamed: 0': 'Time'})
            if 'primaryFault' in data.columns:
                data = data.rename(columns={'primaryFault': 'labels'})
            else:
                data['labels'] = data['labels'] = [0]*data.shape[0]

            if 'secondaryFault' in data.columns:
                data = data.rename(columns={'secondaryFault': 'secondary'})
            else:
                data['secondary'] = [0]*data.shape[0]
            if 'tertiaryFault' in data.columns:
                data = data.rename(columns={'tertiaryFault': 'tertiary'})
            else:
                data['tertiary'] = [0]*data.shape[0]

            data['clusterLabels'] = [0]*data.shape[0]

            sensors = data.columns[1:len(data.columns)-2]
            x_0 = 0
            x_1 = data.shape[0]

            layout = go.Layout(xaxis=dict(range=[x_0, x_1]))

            return 'Fault Labelling: ' + filename, sensors, [sensors[0]], sensors, {'layout': layout}, sensors[0], sensors, sensors[1], sensors, sensors[2], sensors
        else:
            raise PreventUpdate
    except Exception as e:
        raise PreventUpdate

# This function updates the replace / add labels button


@app.callback(Output('replaceAddLabels', 'value'),
              Input('replaceAddLabels', 'value'),
              )
def updateAddLabels(contents):
    if contents == ['Replace Labels'] or contents == ['Replace Labels', 'Add Label'] or contents == ['Add Label', 'Replace Labels']:
        return ['Replace Labels']
    else:
        return ['Add Label']

# This function updates the neural network with new training data


@app.callback(Output('mainGraph', 'figure', allow_duplicate=True),
              Input(Components.uploadTrainingData, 'contents'),
              prevent_initial_call=True)
def updateNeuralNetwork(contents):
    global classifierNeuralNetwork
    try:
        if contents is not None:
            trainingData = pd.DataFrame()
            for i in range(0, len(contents)):
                # for each csv, split & decode data
                contentType, content = contents[i].split(',')
                decoded = io.StringIO(
                    base64.b64decode(content).decode('utf-8'))
                trainingData = trainingData._append(pd.read_csv(decoded))

            # Drop unawanted columns, as a precuation
            if 'commentMessage' in trainingData.columns and 'commentTime' in trainingData.columns and 'commentUser' in trainingData.columns:
                trainingData.drop(
                    columns=['commentMessage', 'commentTime', 'commentUser'], inplace=True)

            if 'Unnamed: 0' in trainingData.columns:
                trainingData.drop(columns=['Unnamed: 0'], inplace=True)

            if 'Time' in trainingData.columns:
                trainingData.drop(columns=['Time'], inplace=True)

            if 'secondaryFault' in trainingData.columns:
                trainingData.drop(columns=['secondaryFault'], inplace=True)

            if 'tertiaryFault' in trainingData.columns:
                trainingData.drop(columns=['tertiaryFault'], inplace=True)

            if 'labels' not in trainingData.columns and 'primaryFault' not in trainingData.columns:
                raise PreventUpdate

            classifierNeuralNetwork = trainNeuralNetwork(trainingData)
            print('neural network saved')
            classifierNeuralNetwork.save(
                "FaultLabeller/NeuralNetworks/multiclassNeuralNetwork")
        #  dont update the callabck
        raise PreventUpdate
    except Exception as e:
        raise PreventUpdate

# This function trains the autoencoder, using the file already in the software


@app.callback(Output('mainGraph', 'figure'),
              Input(Components.uploadNewAutoencoder, 'n_clicks'),
              prevent_initial_call=True
              )
def updateAutoencoder(n_clicks):
    try:
        global autoencoderNeuralNetwork

        if n_clicks is not None:
            trainingData = data.iloc[:, :]
            trainingData.drop(
                columns=['labels', 'Time', 'clusterLabels', 'secondary', 'tertiary'], inplace=True)

            autoencoder = createAutoencoder(trainingData)

            autoencoder.save("FaultLabeller/NeuralNetworks/autoencoder")
            autoencoderNeuralNetwork = autoencoder

        raise PreventUpdate
    except Exception as e:

        raise PreventUpdate

# This function display the dropdowns for colouring the data after the data has been autolabelled


@app.callback(
    Output('ClusterColourContainer', 'children'),
    Input('startAutoLabel', 'n_clicks'),
)
def autoLabelOptions(startAutoLabelClicks):

    # if startAutoLabelClicks == None or startAutoLabelClicks == 0 or data.empty:
    #     raise PreventUpdate
    dropdowns = []
    if 'clusterLabels' in data.columns:
        for i in range(len(set(data['clusterLabels']))):
            dropdowns.append(
                dcc.Markdown('Area ' + str(i+1))
            )
            dropdowns.append(
                dcc.Dropdown(
                    style={'display': 'block', 'width': 200,
                           'background-color': Styles.greyColours[i][0]},
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
    else:
        for i in range(11):
            dropdowns.append(
                dcc.Dropdown(
                    id=f'dropdown-{i}',
                )
            )
    dropdowns.append(
        html.Button('Confirm Labels', id='colorNow', style={
                    'fontSize': 20, 'align-self': 'center', 'font-weight': 'bold', 'margin': 20})
    )
    return dropdowns

# This function colours the graph with the users labels


@app.callback(
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
    State('mainGraph', 'figure'),
    State('mainGraph', 'relayoutData'),
    State('switchView', 'n_clicks'),
    State('alert2div', 'style'),
    State('ClusterColourContainer', 'style'),
    prevent_initial_call=True,
)
def colorLabels(colorNow, area0, area1, area2, area3, area4, area5, area6, area7, area8, area9, figure, relayoutData, switchView, alert2div, clusterColour):
    if colorNow == None or colorNow == 0:
        raise PreventUpdate

    else:
        global x_0
        global x_1
        alert2div['display'] = 'none'
        alertMessage = ''
        if (switchView is None):
            switchView = 0
        areas = [area0, area1, area2, area3, area4,
                 area5, area6, area7, area8, area9]
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
        testAccuracy(data.loc[:, 'labels'])

        data['clusterLabels'] = [0]*data.shape[0]
        ClusterColourContainer = {'display': 'none'}

        return 0, ClusterColourContainer, alert2div, alertMessage

# this function allows the user to close alerts


@app.callback(
    Output('alert1div', 'style', allow_duplicate=True),
    Output('alert2div', 'style', allow_duplicate=True),
    Output('closeAlert1', 'n_clicks'),
    Output('closeAlert2', 'n_clicks'),
    Input('closeAlert1', 'n_clicks'),
    Input('closeAlert2', 'n_clicks'),
    State('alert1div', 'style'),
    State('alert2div', 'style'),
    prevent_initial_call=True
)
def closeAlerts(alert1click, alert2click, alert1style, alert2style):
    try:
        if alert1click == None:
            alert1click = 0
        if alert2click == None:
            alert2click = 0

        if alert1click == 1:
            alert1style['display'] = 'None'

        if alert2click == 1:
            alert2style['display'] = 'None'

        return alert1style, alert2style, 0, 0
    except Exception as e:
        raise PreventUpdate


# This callback tells the user what data point has been clicked
@app.callback(
    Output('alert1div', 'style', allow_duplicate=True),
    Output('alert1', 'children'),
    [Input('mainGraph', 'clickData')],
    State('switchView', 'n_clicks'),
    State('alert1div', 'style'),
    State('alert1', 'children'),
    prevent_initial_call=True
)
def update_textbox(click_data, switchViewClicks, alertstyle, alert):
    try:
        if (switchViewClicks == None):
            switchViewClicks = 0

        if click_data is None:
            raise PreventUpdate
        alertstyle['style'] = 'none'
        labels = ['Unlabelled', 'No Fault', 'Fault 1', 'Fault 2', 'Fault 3',
                  'Fault 4', 'Fault 5', 'Fault 6', 'Fault 7', 'Fault 8', 'Fault 9', 'Fault 10']

        clickedPoint = round(click_data['points'][0]['pointNumber'])
        label = data['labels'][clickedPoint]
        alertstyle['display'] = 'flex'
        alert = 'This point (t = ' + str(clickedPoint) + \
            ') is labelled as ', str(labels[label])

        return alertstyle, alert
    except Exception as e:
        raise PreventUpdate


#  This function updates which sensors are used for auto-labelling
@app.callback(
    Output('sensor-checklist', 'value', allow_duplicate=True),
    Output('select-all', 'n_clicks'),
    Output('deselect-all', 'n_clicks'),
    Output('graphSensors', 'n_clicks'),
    Input('select-all', 'n_clicks'),
    Input('deselect-all', 'n_clicks'),
    Input('graphSensors', 'n_clicks'),
    State(Components.sensorDropdown, 'value'),
    State('switchView', 'n_clicks'),
    State(Components.xAxis_dropdown_3D, 'value'),
    State(Components.yAxis_dropdown_3D, 'value'),
    State(Components.zAxis_dropdown_3D, 'value'),
    prevent_initial_call=True
)
def selectDeselectAll(selectClicks, deselectClicks, graphSensors, sensorDropdown, switchView, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D):
    try:
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
    except Exception as e:
        raise PreventUpdate

#  This function shows and hides clustering paratmeters


@app.callback(
    Output(Components.K, 'style'),
    Output(Components.reducedSize, 'style'),
    Output(Components.AI_text7, 'style'),
    Output(Components.AI_text11, 'style'),
    Output('epsilon', 'style'),
    Output('minVal', 'style'),
    Output(Components.uploadTrainingData, 'style'),
    Output('sensor-checklist', 'value', allow_duplicate=True),
    Output(Components.uploadNewAutoencoder, 'style'),
    Output('switchRepresentation', 'style'),

    Input(Components.clusterMethod, 'value'),
    Input(Components.reductionMethod, 'value'),
    Input('switchView', 'n_clicks'),
    State(Components.uploadTrainingData, 'style'),
    State('sensor-checklist', 'value'),
    State('sensor-checklist', 'options'),
    State(Components.uploadNewAutoencoder, 'style'),
    State('switchRepresentation', 'style'),
    prevent_initial_call=True
)
def autoLabelStyles(clusterMethod, reductionMethod, switchView, uploadTrainingData, sensorChecklistValues, sensorChecklistOptions, uploadNewAutoencoder, switchRepresentation):
    try:
        K_style = {'display': 'none'}
        kMeansMarkdown = {'display': 'none'}
        reducedStyle_style = {'display': 'none'}
        reducedSizeMarkdown = {'display': 'none'}
        epsStyle = {'display': 'none'}
        minValStyle = {'display': 'none'}
        uploadTrainingData['display'] = 'none'
        uploadNewAutoencoder['display'] = 'none'
        switchRepresentation['display'] = 'none'

        if sensorChecklistValues == None:
            sensorChecklistValues = []

        global data
        if data.empty:
            raise PreventUpdate

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

        if (clusterMethod == 'Neural Network (Supervised)'):
            uploadTrainingData['display'] = 'block'
            sensorChecklistValues = sensorChecklistOptions

        if (reductionMethod == 'PCA'):
            reducedStyle_style = {'display': 'block', 'align-self': 'center',
                                  'width': '100%', 'height': '90%', 'fontSize': 20}
            reducedSizeMarkdown = {'display': 'block',
                                   'margin-left': 10, 'width': '50%'}

        if (reductionMethod == 'Auto-encoding'):
            sensorChecklistValues = sensorChecklistOptions
            uploadNewAutoencoder['display'] = 'block'

        if switchView % 3 == 1 or switchView % 3 == 2:
            switchRepresentation['display'] = 'block'

        return K_style, reducedStyle_style, reducedSizeMarkdown, kMeansMarkdown, epsStyle, minValStyle, uploadTrainingData, sensorChecklistValues, uploadNewAutoencoder, switchRepresentation
    except Exception as e:
        raise PreventUpdate

# This function finds the optimal value of minPts and epsilon, dependent on the user selected parameters


@app.callback(
    Output(Components.minPtsSlider, 'value'),
    Output(Components.epsSlider, 'value'),
    Input(Components.clusterMethod, 'value'),
    Input('sensor-checklist', 'value'),
    Input('reducedSize', 'value'),
    Input(Components.reductionMethod, 'value')
)
def DBSCAN_parameterSelection(clusterMethod, sensorChecklist, reducedSize, reductionMethod):
    try:
        if clusterMethod == 'DBSCAN' and sensorChecklist != [] and reductionMethod == 'PCA' and reducedSize != None:
            df = data.loc[:, sensorChecklist]
            df = performPCA(df, 10)
            eps = findKneePoint(df, 10)
            return 10, eps
        elif clusterMethod == 'DBSCAN' and sensorChecklist != [] and reductionMethod == 'None':
            df = data.loc[:, sensorChecklist]
            eps = findKneePoint(df, 10)
        elif clusterMethod == 'DBSCAN' and sensorChecklist != [] and reductionMethod == 'Auto-encoding':
            df = data.iloc[:, :]
            df = autoencoderNeuralNetwork.predict(df)
            eps = findKneePoint(df, 10)

            return 10, eps
        else:
            raise PreventUpdate
    except Exception as e:
        raise PreventUpdate

# This function adds comments


@app.callback(
    Output("commentModal", "style"),
    Output("commentModal", 'children'),
    Output("addComment", "n_clicks"),
    Output('closeComments', 'n_clicks'),
    Output('open-modal', 'n_clicks'),

    Input("open-modal", "n_clicks"),
    Input('closeComments', 'n_clicks'),
    Input("addComment", "n_clicks"),
    State("commentInput", 'value'),
    State("usernameInput", 'value'),
    State("commentModal", "style"),
)
def toggle_modal(openModal, closeModal, addComments, commentInput, usernameInput, commentModalStyle):
    try:
        if openModal == None:
            openModal = 0
        if closeModal == None:
            closeModal = 0

        if openModal == 1:
            commentModalStyle['display'] = 'block'
        if closeModal == 1:
            commentModalStyle['display'] = 'none'

        global comments
        time = datetime.datetime.now().strftime("%H:%M")

        if (addComments == 1):
            comments = comments._append({'commentTime': time,
                                        'commentUser': usernameInput, 'commentMessage': commentInput}, ignore_index=True)

        modalChidren = [
            html.Div(style={'position': 'relative'}, children=[
                dcc.Markdown("Comments", style={
                    'fontWeight': 'bold', 'fontSize': 30}),
                html.Button("X", id='closeComments', style={'position': 'absolute', 'right': 10, 'top': 10})]
            )]

        for i in range(comments.shape[0]):
            modalChidren.append(
                html.Div(style={'flex-direction': 'row', 'display': 'flex', 'justify-content': 'space-evenly'},  children=[
                    dcc.Markdown(comments.iloc[i, 0], style={
                        'width': '15%', 'font-size': 20}),
                    dcc.Markdown(comments.iloc[i, 1], style={
                        'width': '20%', 'font-size': 20}),
                    dcc.Markdown(comments.iloc[i, 2], style={
                        'width': '65%', 'font-size': 20}),
                ]),)

        modalChidren.append(html.Div(
            html.Div(children=[
                dcc.Input(id='commentInput', type='text',
                          value='Comment', style={'font-size': 20}),
                dcc.Input(id='usernameInput', type='text',
                          value='Name', style={'font-size': 20}),
                html.Button("Add Comment", id='addComment', style={'font-size': 20, 'font-weight': 'bold'})]
            ),
        ),)

        return commentModalStyle, modalChidren, 0, 0, 0
    except Exception as e:
        raise PreventUpdate

# This function performs the bulk of functionality: everything that uses the main graph as an output do with main graph


@app.callback(
    Output('mainGraph', 'figure', allow_duplicate=True),
    Output('labelButton', 'children'),
    Output('removeLabels', 'n_clicks'),
    Output('findPrev', 'n_clicks'),
    Output('findNext', 'n_clicks'),
    Output(Components.stat1, 'children'),
    Output(Components.stat2, 'children'),
    Output(Components.stat3, 'children'),
    Output('alert2div', 'style', allow_duplicate=True),
    Output('alert2', 'children', allow_duplicate=True),
    Output(Components.sensorDropdown, 'style'),
    Output('startAutoLabel', 'n_clicks'),
    Output('ClusterColourContainer', 'style'),
    Output('xAxisDropdownContainer', 'style'),
    Output('yAxisDropdownContainer', 'style'),
    Output('zAxisDropdownContainer', 'style'),

    Input(Components.sensorDropdown, 'value'),
    State(Components.labelDropdown, 'value'),
    Input('switchView', 'n_clicks'),
    Input('labelButton', 'n_clicks'),
    Input('removeLabels', 'n_clicks'),
    Input('findPrev', 'n_clicks'),
    Input('findNext', 'n_clicks'),
    Input(Components.faultFinder, 'value'),
    Input('mainGraph', 'clickData'),
    Input(Components.xAxis_dropdown_3D, 'value'),
    Input(Components.yAxis_dropdown_3D, 'value'),
    Input(Components.zAxis_dropdown_3D, 'value'),
    Input('startAutoLabel', 'n_clicks'),
    Input('colorNow', 'n_clicks'),
    Input('switchRepresentation', 'n_clicks'),

    State('sensor-checklist', 'value'),
    State(Components.clusterMethod, 'value'),
    State(Components.reductionMethod, 'value'),
    State('mainGraph', 'relayoutData'),
    State(Components.K, 'value'),
    State('reducedSize', 'value'),
    State(Components.epsSlider, 'value'),
    State(Components.minPtsSlider, 'value'),
    State('alert2div', 'style'),
    State('alert2', 'children'),
    State('replaceAddLabels', 'value'),
    prevent_initial_call=True,

)
def updateGraph(sensorDropdown, labelDropdown, switchViewButtonClicks, labelButtonClicks, removeLabelClick, findPrevClicked, findNextClicked, faultFinder, clickData, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, newAutoLabel,  colorNow, switchRepresentation, sensorChecklist, clusterMethod, reductionMethod, relayoutData, K, reducedSize, eps, minVal, alert2div, alert2, replaceAddLabel):
    try:
        global shapes
        global colours
        global x_0
        global x_1
        global navigationPoint
        global clickedPoint
        global classifierNeuralNetwork
        global autoencoderNeuralNetwork
        global data

        start_time = time.time()

        if data.empty:
            raise PreventUpdate

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

        if (switchRepresentation == None):
            switchRepresentation = 0

        # Set default output values
        labelButtonTitle = 'New Label'
        ClusterColourContainer = {"display": "none"}
        xAxis_dropdown_3D_style = {"display": "none"}
        yAxis_dropdown_3D_style = {"display": "none"}
        zAxis_dropdown_3D_style = {"display": "none"}
        alert2div['display'] = 'none'

        # Take note of initial x0 and x1 values
        if relayoutData and 'xaxis.range[0]' in relayoutData.keys():
            x_0 = relayoutData.get('xaxis.range[0]')
            x_1 = relayoutData.get('xaxis.range[1]')

        if (switchViewButtonClicks % 3 == 0):  # i.e. time graph

            # ADD DATA
            selectData = []
            if sensorDropdown != None:
                if sensorDropdown != []:
                    for i in range(len(sensorDropdown)):
                        selectData.append(go.Scatter(
                            y=data.loc[:, sensorDropdown[i]], name=sensorDropdown[i], yaxis='y' + str(i+1), opacity=1-0.2*i))

            if clickData is not None and 'points' in clickData:
                clickedPoint = clickData['points'][0]['x']
    #
            if (labelButtonClicks % 2 == 0):  # i.e. a label has been added

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

                    if replaceAddLabel == ['Add Label'] and labelDropdown != 0 and labelDropdown != 1:
                        # if (1):
                        primarySet = set(
                            data['labels'][x0:x1].values.flatten())
                        secondarySet = set(
                            data['secondary'][x0:x1].values.flatten())
                        tertiarySet = set(
                            data['tertiary'][x0:x1].values.flatten())

                        # if all values are 0 or 1 in primary, repalce primary labels and remove secondary/teriary
                        if all(val == 0 or val == 1 or val == labelDropdown for val in primarySet):
                            # all values are 0 or 1
                            data['labels'][x0:x1] = [labelDropdown] * (x1-x0)
                        # if there is already a fault in primary, then check secondary does not already cotnans a fault
                        elif all(val == 0 or val == labelDropdown for val in secondarySet):
                            data['secondary'][x0:x1] = [
                                labelDropdown] * (x1-x0)
                        #  if theres already a fault in secondary, add it to tertriary
                        elif all(val == 0 for val in tertiarySet):
                            data['tertiary'][x0:x1] = [labelDropdown] * (x1-x0)
                        #  else, all the labels are full and an error should be thrown
                        else:
                            alert2div['display'] = 'flex'
                            alert2 = 'You have placed the maximum number of fault labels.'
                    else:
                        data['labels'][x0:x1] = [labelDropdown] * (x1-x0)
                        data['secondary'][x0:x1] = [0] * (x1-x0)
                        data['tertiary'][x0:x1] = [0] * (x1-x0)

            elif (labelButtonClicks % 2 == 1):   # i.e. a label is about to be added

                dragMode = 'select'
                labelButtonTitle = "Confirm Label"

            if (findNextClicked == 1):  # i.e. user is navigating
                target = 0
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

                if (int(navigationPoint) == len(data['labels'])):
                    # Create an alert to informt that there are no furher ponts
                    alert2div['display'] = 'flex'
                    alert2 = 'You have reached the end of the data.'
                else:
                    start = -1
                    end = -1
                    for i in range(int(navigationPoint), len(data['labels'])):
                        if (data['labels'][i] == target):
                            start = i
                            for j in range(i, len(data['labels'])):
                                if (data['labels'][j] != target):
                                    end = j
                                    navigationPoint = str(end)
                                    break
                            if (end == -1):
                                end = len(data['labels'])
                                navigationPoint = str(end)
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

            if (findPrevClicked == 1):  # ie. user is navgating back

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

                if (int(navigationPoint) == 0):
                    # Create an alert to informt that there are no furher ponts
                    alert2div['display'] = 'flex'
                    alert2 = 'You have reached the start of the data'

                else:
                    start = -1
                    end = -1

                    for i in range(int(navigationPoint)-1, 0, -1):
                        if (data['labels'][i] == target):
                            end = i
                            start = -1
                            for j in range(i, 0, -1):
                                if (data['labels'][j] != target):
                                    start = j
                                    navigationPoint = str(start)
                                    break
                            if (start == -1):
                                start = 0
                                navigationPoint = str(start)
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

            if (newAutoLabel == 1):  # i.e time to perform autoabelling

                if (sensorChecklist == [] or sensorChecklist == None):
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
                        df = data.loc[:, :]
                        if 'Unnamed: 0' in df.columns:
                            df.drop(columns=['Unnamed: 0'], inplace=True)
                        if 'Time' in df.columns:
                            df.drop(columns=['Time'], inplace=True)
                        if 'secondary' in df.columns:
                            df.drop(columns=['secondary'], inplace=True)
                        if 'tertiary' in df.columns:
                            df.drop(columns=['tertiary'], inplace=True)
                        if 'labels' in df.columns:
                            df.drop(columns=['labels'], inplace=True)
                        if 'clusterLabels' in df.columns:
                            df.drop(columns=['clusterLabels'], inplace=True)

                        latentSpace = autoencoderNeuralNetwork.predict(df)

                        df = pd.DataFrame(data=latentSpace)

                    if (clusterMethod == 'K Means'):
                        if (K == None or K < 0):
                            alert2div['display'] = 'flex'
                            alert2 = 'Wrong value input for K Means. Clustering has failed.'
                        elif (K > 10 or K <= 1):
                            alert2div['display'] = 'flex'
                            alert2 = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performKMeans(df, K)
                            ClusterColourContainer = {
                                'display': 'block', 'width': 200, 'padding': 20}

                    elif (clusterMethod == 'DBSCAN'):
                        if (eps == None or minVal == None):
                            alert2div['display'] = 'flex'
                            alert2 = 'Incorrect parameter for eps or min points.'
                        else:

                            n = len(sensorChecklist)
                            temp = performDBSCAN(df, eps, minVal)
                            if len(list(set(temp))) >= 10:
                                alert2div['display'] = 'flex'
                                alert2 = 'DBSCAN produced too many clusters. Try decreasing epsilon or increasing min points.'
                            elif (len(list(set(temp))) == 1):
                                alert2div['display'] = 'flex'
                                alert2 = 'DBSCAN produced only outliers. Try increasing epsilon or decreasing min points.'
                            else:
                                data['labels'] = [0]*data.shape[0]
                                data['clusterLabels'] = temp

                                ClusterColourContainer = {
                                    'display': 'block', 'width': 200, 'padding': 20}
                    elif (clusterMethod == 'Neural Network (Supervised)'):
                        df = data.iloc[:, :]
                        df.drop(columns=['clusterLabels'], inplace=True)
                        df.drop(columns=['labels'], inplace=True)
                        df.drop(columns=['secondary'], inplace=True)
                        df.drop(columns=['tertiary'], inplace=True)
                        df.drop(columns=['Time'], inplace=True)

                        data['labels'] = useNeuralNetwork(
                            df, classifierNeuralNetwork)
                        testAccuracy(data.loc[:, 'labels'])
                        data['clusterLabels'] = [0]*data.shape[0]
                    x_0 = 0
                    x_1 = data.shape[0]

            if (removeLabelClick == 1):
                data['labels'] = [0]*data.shape[0]
                data['clusterLabels'] = [0]*data.shape[0]
                data['secondary'] = [0]*data.shape[0]
                data['tertiary'] = [0]*data.shape[0]

            # Go through labels and shown all the shapes
            shapes = []
            x0 = 0
            x1 = x0
            x0_sec = 0
            x1_sec = x0_sec
            x0_ter = 0
            x1_ter = x0_ter

            if 'labels' in data.columns:
                for i in range(1, len(data['labels'])):

                    if data['labels'][i] != data['labels'][i-1]:

                        x1 = i

                        shapes.append({
                            'type': 'rect',
                            'x0': x0,
                            'x1': x1,
                            'y0': 0,
                            'y1': 0.05,
                            'fillcolor': Styles.colours[int(data.loc[x0, 'labels'])][0],
                            'yref': 'paper',
                            'opacity': 1,
                            'name': str(data['labels'][x0])
                        },)

                        x0 = i
                    if data['secondary'][i] != data['secondary'][i-1]:

                        x1_sec = i
                        if data.loc[x0_sec, 'secondary'] != 0:
                            shapes.append({
                                'type': 'rect',
                                'x0': x0_sec,
                                'x1': x1_sec,
                                'y0': 0.05,
                                'y1': 0.075,
                                'fillcolor': Styles.colours[int(data.loc[x0_sec, 'secondary'])][0],
                                'yref': 'paper',
                                'opacity': 1,
                                'name': str(data['secondary'][x0_sec])
                            },)

                        x0_sec = i
                    if data['tertiary'][i] != data['tertiary'][i-1]:

                        x1_ter = i
                        if data.loc[x0_ter, 'tertiary'] != 0:
                            shapes.append({
                                'type': 'rect',
                                'x0': x0_ter,
                                'x1': x1_ter,
                                'y0': 0.075,
                                'y1': 0.1,
                                'fillcolor': Styles.colours[int(data.loc[x0_ter, 'tertiary'])][0],
                                'yref': 'paper',
                                'opacity': 1,
                                'name': str(data['tertiary'][x0_ter])
                            },)

                        x0_ter = i

                shapes.append({
                    'type': 'rect',
                    'x0': x0,
                    'x1': len(data['labels']),
                    'y0': 0,
                    'y1': 0.05,
                    'fillcolor': Styles.colours[int(data.loc[x0, 'labels'])][0],
                    'yref': 'paper',
                    'opacity': 1,
                    'name': str(data['labels'][x0])
                },)
                if data.loc[x0_sec, 'secondary'] != 0:
                    shapes.append({
                        'type': 'rect',
                        'x0': x0_sec,
                        'x1': len(data['secondary']),
                        'y0': 0.05,
                        'y1': 0.075,
                        'fillcolor': Styles.colours[int(data.loc[x0_sec, 'secondary'])][0],
                        'yref': 'paper',
                        'opacity': 1,
                        'name': str(data['secondary'][x0_sec])
                    },)
                if data.loc[x0_ter, 'tertiary'] != 0:
                    shapes.append({
                        'type': 'rect',
                        'x0': x0_ter,
                        'x1': len(data['tertiary']),
                        'y0': 0.075,
                        'y1': 0.1,
                        'fillcolor': Styles.colours[int(data.loc[x0_ter, 'tertiary'])][0],
                        'yref': 'paper',
                        'opacity': 1,
                        'name': str(data['tertiary'][x0_ter])
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
                                'fillcolor': Styles.greyColours[data['clusterLabels'][x0]][0],
                                'yref': 'paper',
                                'opacity': 0.9,
                                'name': 'area'+str(data['clusterLabels'][x0])
                            },)

                            x0 = i

                    shapes.append({
                        'type': 'rect',
                        'x0': x0,
                        'x1': len(data['labels']),
                        'y0': 0,
                        'y1': 0.05,
                        'fillcolor': Styles.greyColours[data['clusterLabels'][x0]][0],
                        'yref': 'paper',
                        'opacity': 0.9,
                        'name': 'area'+str(data['clusterLabels'][x0])
                    },)

            layout = go.Layout(legend={'x': 0, 'y': 1.2}, xaxis=dict(range=[x_0, x_1]),  dragmode=dragMode, yaxis=dict(fixedrange=True, title='Sensor Value', color='blue'), yaxis2=dict(
                fixedrange=True, overlaying='y', color='orange', side='right'), yaxis3=dict(fixedrange=True, overlaying='y', color='green', side='left', position=0.001,), yaxis4=dict(fixedrange=True, overlaying='y', color='red', side='right'), shapes=shapes)

            if clickedPoint is not None:
                selectData.append(
                    go.Line(x=[clickedPoint, clickedPoint], y=[0, data.loc[:, sensorDropdown[0]].max()], name='Selected Point', line=dict(color='black')))

            fig = {'data': selectData, 'layout': layout, }

            sensorDropdownStyle = {'display': 'block',
                                   'fontSize': 20, 'margin': 10}

        if (switchViewButtonClicks % 3 == 1):  # i.e. 2D Scatter

            xAxis_dropdown_3D_style = {"display": "flex"}
            yAxis_dropdown_3D_style = {"display": "flex"}

            sensorDropdownStyle = {'display': 'none'}

            if clickData is not None and 'points' in clickData:

                clickedPoint = clickData['points'][0]['pointNumber']

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

            if (newAutoLabel == 1):  # i.e time to perform autoabelling

                if (sensorChecklist == [] or sensorChecklist == None):
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
                        df = data.iloc[:, :]
                        if 'Unnamed: 0' in df.columns:
                            df.drop(columns=['Unnamed: 0'], inplace=True)
                        if 'Time' in df.columns:
                            df.drop(columns=['Time'], inplace=True)
                        if 'secondary' in df.columns:
                            df.drop(columns=['secondary'], inplace=True)
                        if 'tertiary' in df.columns:
                            df.drop(columns=['tertiary'], inplace=True)
                        if 'labels' in df.columns:
                            df.drop(columns=['labels'], inplace=True)
                        if 'clusterLabels' in df.columns:
                            df.drop(columns=['clusterLabels'], inplace=True)

                        latentSpace = autoencoderNeuralNetwork.predict(
                            df)

                        df = pd.DataFrame(data=latentSpace)

                    if (clusterMethod == 'K Means'):
                        if (K == None or K < 0):
                            alert2div['display'] = 'flex'
                            alert2 = 'Wrong value input for K Means. Clustering has failed.'
                        elif (K > 10 or K <= 1):
                            alert2div['display'] = 'flex'
                            alert2 = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performKMeans(df, K)
                            ClusterColourContainer = {
                                'display': 'block', 'width': 200, 'padding': 20}

                    elif (clusterMethod == 'DBSCAN'):
                        if (eps == None or minVal == None):
                            alert2div['display'] = 'flex'
                            alert2 = 'Incorrect parameter for eps or min points.'
                        else:
                            temp = performDBSCAN(df, eps, minVal)
                            if len(list(set(temp))) >= 10:
                                alert2div['display'] = 'flex'
                                alert2 = 'DBSCAN produced too many clusters. Try decreasing epsilon or increasing min points.'
                            elif (len(list(set(temp))) == 1):
                                alert2div['display'] = 'flex'
                                alert2 = 'DBSCAN produced only outliers. Try increasing epsilon or decreasing min points.'
                            else:
                                data['labels'] = [0]*data.shape[0]
                                data['clusterLabels'] = performDBSCAN(
                                    df, eps, minVal)
                                ClusterColourContainer = {
                                    'display': 'block', 'width': 200, 'padding': 20}
                    elif (clusterMethod == 'Neural Network (Supervised)'):
                        df = data.iloc[:, :]
                        df.drop(columns=['clusterLabels'], inplace=True)
                        df.drop(columns=['labels'], inplace=True)
                        df.drop(columns=['secondary'], inplace=True)
                        df.drop(columns=['tertiary'], inplace=True)
                        df.drop(columns=['Time'], inplace=True)

                        data['labels'] = useNeuralNetwork(
                            df, classifierNeuralNetwork)
                        data['clusterLabels'] = [0]*data.shape[0]

                    x_0 = 0
                    x_1 = data.shape[0]

                    selectData = [go.Scatter(
                        y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
                                                                     xAxis_dropdown_3D], text=data.loc[:, 'clusterLabels'], mode='markers', marker={'color': [Styles.greyColours[val][0] for val in data['clusterLabels']], })]

                    if (clusterMethod == 'Neural Network (Supervised)'):
                        selectData = [go.Scatter(
                            y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
                                                                         xAxis_dropdown_3D], text=data.loc[:, 'labels'], mode='markers', marker={'color': [Styles.colours[val][0] for val in data['clusterLabels']], })]
            else:
                if (removeLabelClick == 1):
                    data['labels'] = [0]*data.shape[0]
                    data['clusterLabels'] = [0]*data.shape[0]
                selectData = [go.Scatter(
                    y=data.loc[:, yAxis_dropdown_3D], x=data.loc[:,
                                                                 xAxis_dropdown_3D], text=data.loc[:, 'labels'], mode='markers', marker={'color': [Styles.colours[int(val)][0] for val in data['labels']], })]
            layout = go.Layout(dragmode=dragMode, yaxis=dict(
                title=yAxis_dropdown_3D), xaxis=dict(
                title=xAxis_dropdown_3D))

            fig = {'data': selectData, 'layout': layout}

            if (switchRepresentation % 2 == 1):
                fig = px.scatter(data, x=xAxis_dropdown_3D,
                                 y=yAxis_dropdown_3D, color='Time',)

            if clickedPoint is not None:
                selectData.append(go.Scatter(x=[data[xAxis_dropdown_3D][clickedPoint]], y=[
                    data[yAxis_dropdown_3D][clickedPoint]],  marker=dict(color='black', size=40)))

        if (switchViewButtonClicks % 3 == 2):
            #  3D SCATTER PLOT

            xAxis_dropdown_3D_style = {"display": "flex"}
            yAxis_dropdown_3D_style = {"display": "flex"}
            zAxis_dropdown_3D_style = {"display": "flex"}

            if clickData is not None and 'points' in clickData:

                clickedPoint = clickData['points'][0]['pointNumber']

            if (newAutoLabel == 1):

                selectData = []

                if (sensorChecklist == [] or sensorChecklist == None):
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
                        df = data.iloc[:, :]
                        if 'Unnamed: 0' in df.columns:
                            df.drop(columns=['Unnamed: 0'], inplace=True)
                        if 'Time' in df.columns:
                            df.drop(columns=['Time'], inplace=True)
                        if 'secondary' in df.columns:
                            df.drop(columns=['secondary'], inplace=True)
                        if 'tertiary' in df.columns:
                            df.drop(columns=['tertiary'], inplace=True)
                        if 'labels' in df.columns:
                            df.drop(columns=['labels'], inplace=True)
                        if 'clusterLabels' in df.columns:
                            df.drop(columns=['clusterLabels'], inplace=True)
                        latentSpace = autoencoderNeuralNetwork.predict(df)

                        df = pd.DataFrame(data=latentSpace)

                    if (clusterMethod == 'K Means'):
                        if (K == None or K < 0):
                            alert2div['display'] = 'flex'
                            alert2 = 'Wrong value input for K Means. Clustering has failed.'
                        elif (K > 10 or K <= 1):
                            alert2div['display'] = 'flex'
                            alert2 = 'Select a value between 1 and 10 for K.'
                        else:
                            data['labels'] = [0]*data.shape[0]
                            data['clusterLabels'] = performKMeans(df, K)
                            ClusterColourContainer = {
                                'display': 'block', 'width': 200, 'padding': 20}

                    elif (clusterMethod == 'DBSCAN'):
                        if (eps == None or minVal == None):
                            alert2div['display'] = 'flex'
                            alert2 = 'Incorrect parameter for eps or min points.'
                        else:
                            n = len(sensorChecklist)
                            temp = performDBSCAN(df, eps, minVal)
                            if len(list(set(temp))) >= 10:
                                alert2div['display'] = 'flex'
                                alert2 = 'DBSCAN produced too many clusters. Try decreasing epsilon or increasing min points.'
                            elif (len(list(set(temp))) == 1):
                                alert2div['display'] = 'flex'
                                alert2 = 'DBSCAN produced only outliers. Try increasing epsilon or decreasing min points.'
                            else:
                                data['labels'] = [0]*data.shape[0]
                                data['clusterLabels'] = performDBSCAN(
                                    df, eps, minVal)
                                ClusterColourContainer = {
                                    'display': 'block', 'width': 200, 'padding': 20}
                    elif (clusterMethod == 'Neural Network (Supervised)'):
                        df = data.iloc[:, :]
                        df.drop(columns=['clusterLabels'], inplace=True)
                        df.drop(columns=['labels'], inplace=True)
                        df.drop(columns=['secondary'], inplace=True)
                        df.drop(columns=['tertiary'], inplace=True)
                        df.drop(columns=['Time'], inplace=True)

                        data['labels'] = useNeuralNetwork(
                            df, classifierNeuralNetwork)
                        data['clusterLabels'] = [0]*data.shape[0]

                    x_0 = 0
                    x_1 = data.shape[0]

                    selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:, zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers', marker={
                        'size': 10, 'opacity': 1, 'color': [Styles.greyColours[val][0] for val in data['clusterLabels']], },)]

                    if (clusterMethod == 'Neural Network (Supervised)'):
                        selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:, zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers', marker={
                            'size': 10, 'opacity': 1, 'color': [Styles.colours[val][0] for val in data['labels']], },)]

            else:
                if (removeLabelClick == 1):
                    data['labels'] = [0]*data.shape[0]
                    data['clusterLabels'] = [0]*data.shape[0]
                selectData = [go.Scatter3d(y=data.loc[:, yAxis_dropdown_3D], z=data.loc[:,
                                                                                        zAxis_dropdown_3D], x=data.loc[:, xAxis_dropdown_3D], mode='markers',
                                           marker={
                    'size': 10,
                    'opacity': 1,
                    'color': [Styles.colours[int(val)][0] for val in data['labels']],
                },)]
            if clickedPoint is not None:
                selectData.append(go.Scatter3d(x=[data[xAxis_dropdown_3D][clickedPoint]], y=[
                    data[yAxis_dropdown_3D][clickedPoint]], z=[data[zAxis_dropdown_3D][clickedPoint]], marker=dict(color='black', size=40)))

            sensorDropdownStyle = {'display': 'none'}
            layout = go.Layout(xaxis=dict(
                title=xAxis_dropdown_3D), yaxis=dict(
                title=yAxis_dropdown_3D), )

            fig = {'data': selectData, 'layout': layout}

            if (switchRepresentation % 2 == 1):
                fig = px.scatter_3d(data, x=xAxis_dropdown_3D,
                                    y=yAxis_dropdown_3D, z=zAxis_dropdown_3D, color='Time',)

        stat1 = 'Number of unlabelled data points: '
        stat2 = 'Number of labelled data points: '
        stat3 = 'Number of labels Placed: '

        if 'labels' in data.columns:
            labels = data['labels'].values.tolist()
            n = labels.count(0)
            stat1 += str(n)
            stat2 += str(len(data['labels']) - n)
            n = len(set(labels))
            if 0 in set(labels):
                stat3 += str(len(set(labels))-1)
            else:
                stat3 += str(len(set(labels)))

        end_time = time.time()
        elapsed_time = end_time - start_time
        print("Elapsed time:", elapsed_time, "seconds")

        return fig, labelButtonTitle, 0, 0, 0, stat1, stat2, stat3, alert2div,  alert2, sensorDropdownStyle, 0, ClusterColourContainer, xAxis_dropdown_3D_style, yAxis_dropdown_3D_style, zAxis_dropdown_3D_style
    except Exception as e:
        raise PreventUpdate


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
