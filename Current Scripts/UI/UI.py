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
from Components import title, sensorDropdown, sensorHeader, labelDropdown, stat3, faultFinderHeader, faultFinderText, stat1, stat2, exportName, exportHeader, exportLocation, exportConfirm, AI_header,AI_text1,clusterMethod,AI_text2,reductionMethod, AI_input1, AI_input2, AI_input3,AI_input4
from myFunctions import changeText, updateGraph, performKMeans

app = dash.Dash(__name__)

# DATA
data = pd.read_csv("Data/UpdatedData.csv")
data = data.drop(data.columns[[1, 2, 3]], axis=1) # Remove extra columns
data = data.rename(columns={'Unnamed: 0': 'Time'}) # Rename First Column

# GLOBAL VARIABLES
shapes = [] # An array which stores rectangles, to visualise labels
currentPoint = 0 # The current point, for navigation 
labels = [-1]*data.shape[0]  # -1 for non label, 0 for no fault, 1 for fault 1, 2 for fault 2 etc

# What proportion of the screen is shown
x_0 = 0 
x_1 = 5000

colours = [['blue'], ['green'], ['orange'], [
                'purple'], ['pink'], ['violet'], ['lavender']]

# t is used to switch between time based view and 3D based view
t = None



# Define the layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, blue, #000000)', 'height': '100vh', 'display': 'flex', 'justify-content': 'center', 'flex-direction': 'column', 'align-items': 'center'},children=[
                alert,
                html.Div(style={'height':'5%', 'width':'90%'}, 
                        children=[title]),
                    
                # Top Box
                html.Div(style={'overflow':'scroll', 'width':'90%', 'height':'55%', 'margin':'20px', 'border-radius': '10px', 'padding': '20px', 'background-color': 'white'},
                    children=[
                        # dcc.Markdown(id='latestAction', children='Latest Action: '),
                        html.Button('Switch View', id='switchView'),
                        html.Button(children='Observe Clusters', id='observe-clusters'),
                        mainGraph, 
                        
                        # dcc.Graph(id='tempGraph')
                        # zoom
                        # fourGraphs
                        ]),

                # Bottom boxes
                html.Div(style={'width': '90%', 'height': '35%',  'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin':'20px' },
                    children=[
                        # Box 1
                        html.Div(style={'overflow':'scroll', 'border-radius': '10px', 'width': '24%', 'height': '100%',  'background-color': 'white'}, 
                            children=[
                                sensorHeader, 
                                sensorDropdown,
                                xAxis_dropdown_3D, 
                                yAxis_dropdown_3D, 
                                zAxis_dropdown_3D]),

                        
                        html.Div(style={'width': '24%', 'height': '100%', }, children=[
                            #  Box 2
                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'},
                                children=[
                                    # zoom, 
                                    # pan
                                     
                                    html.Button(children='Start Label', id='labelButton', style={'width':'100%', 'height': 40, 'fontSize': 16}), 
                                    labelDropdown,
                                    html.Button('Remove Label', id='removeLabels', style={'width':'50%'}), 
                                    html.Button('Undo Label', id='undoLabel', style={'width':'50%'}),
                                    html.Button('Change Label', id='changeLabel', style={'width':'50%'}),
                                    html.Button('Move Label', id='moveLabel', style={'width':'50%'}),
                                    ]), 

                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '6%'}),
                            
                            #  Box 3
                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'}, 
                                children=[
                                    html.Div(style={'display':'flex', 'flex-direction':'column', 'justify-content':'center', 'text-align':'left', 'align-items': 'center'},
                                        children=[
                                        faultFinderHeader,
                                        html.Div(style={'flex-direction':'column', 'display':'flex', 'width':'90%'}, children=[
                                            faultFinderText,
                                            faultFinder
                                        ]),
                                        html.Div(style={'flex-direction':'row'}, children=[
                                            html.Button('Prev', id='findPrev'),
                                            html.Button('Next', id='findNext'),
                                        ])
                                        
                                    
                                    # dcc.Markdown('0', id='currentPoint')
                                    ])
                                ]),]),
                            
                            # Box 4
                            html.Div(style={'overflow':'scroll',  'border-radius': '10px','width': '24%', 'height': '100%', 'background-color': 'white'}, 
                                     children=[  
                                            AI_header,
                                            dcc.Markdown('Algorithms: ', style={'fontsize':26, 'fontWeight':'bold'}),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('Clustering Method:'),
                                                clusterMethod,
                                            ]),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('Reduction Method:'),
                                                reductionMethod
                                            ]),
                                            
                                            
                                            dcc.Markdown('Parameters: ', style={'fontsize':26, 'fontWeight':'bold'}),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('K ='),
                                                dcc.Input(type='number')
                                            ]),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('Min Val ='),
                                                dcc.Input(type='number')
                                            ]),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('Epsilon ='),
                                                dcc.Input(type='number')
                                            ]),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('Starting Values ='),
                                                html.Button("Select all", id='select-all'),
                                                html.Button("Deselect all", id='deselect-all'),
                                            ]),
                                            html.Div(style={'width': '100%', 'height': 150, 'overflow':'scroll'}, children=[
                                                    dcc.Checklist(id='sensor-checklist', options=data.columns, inline=True, labelStyle={'width':'33%'})
                                                ]),
                                            html.Div(style={'display':'flex', }, children=[
                                                dcc.Markdown('Reduced Size:'),
                                                dcc.Input(type='number')
                                            ]),
                     
                                            html.Button(children='Start Now', id='startAutoLabel', style={'width':'100%', 'fontSize': 20})
                                            
                                    
                                         ]),

                            # Box 5
                            html.Div(style={'width': '24%', 'height': '100%', 'overflow':'scroll' }, 
                                     children=[    
                                            html.Div(style={'overflow':'scroll','border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'}, 
                                                    children=[
                                                              stat1, stat2, stat3,
                                                              dcc.Markdown('Shape clicked', id='shape-clicked'),
                                                              dcc.Markdown('Points Output:', id='points-output',style={"display":"block"}),]
                                                    ),
                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '6%', }),
                            
                            # Box6
                            html.Div(style={'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white', 'display': 'flex','justify-content': 'center',}, 
                                children=[
                                    html.Div(style={'display':'flex', 'flex-direction':'column', 'justify-content':'center', 'align-items': 'center'},
                                             children=[
                                                 exportHeader,
                                                 dcc.Markdown('file name:'),
                                                 exportName,
                                                 exportConfirm,
                                                 dcc.Download(id="downloadData"),

                
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

    return  values

# This function exports data to a downloadable csv
@app.callback(
    Output('downloadData', 'data'),
    Input(exportConfirm, 'n_clicks'),
    State(exportName, 'value')
)
def exportCSV(clicked, fileName):
    if clicked is None:
        raise PreventUpdate
    
    df = pd.DataFrame(labels)

    csv_filename = fileName + '.csv'
    df.to_csv(csv_filename, index=False)
    return  dcc.send_file(csv_filename)


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
    Input('observe-clusters', 'n_clicks'),

    State('sensor-checklist', 'value'),
    State(clusterMethod, 'value'),
    State(reductionMethod, 'value'),

    State(mainGraph, 'relayoutData'),
)
def updateGraph(sensorDropdown, labelDropdown, switchViewButtonClicks, labelButtonClicks,removeLabelClick, undoLabelClick, findPrevClicked, findNextClicked, faultFinder, clickData, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, newAutoLabel, observeClusterClikcs, sensorChecklist, clusterMethod, reductionMethod, relayoutData ):
    fig = px.line()
    global shapes
    global labels

    if (newAutoLabel == None):
        newAutoLabel = 0

    if (observeClusterClikcs == None):
        observeClusterClikcs = 0
    stat1 =''
    alert = False
    alertMessage = ''

    global x_0
    global x_1
    if relayoutData and 'xaxis.range[0]' in relayoutData.keys():
        x_0 = relayoutData.get('xaxis.range[0]')
        x_1 = relayoutData.get('xaxis.range[1]')
    

    if (switchViewButtonClicks == None):
        switchViewButtonClicks = 0

    if (findNextClicked == None):
        findNextClicked = 0


    if (switchViewButtonClicks % 2 == 0):
        max = 1
        selectData = []
        for i in range(len(sensorDropdown)):
            name = sensorDropdown[i]
            yaxis = 'y' + str(i+1)
            
            if (data.loc[:, sensorDropdown[i]].max() > max):
                 max = data.loc[:, sensorDropdown[i]].max()

            selectData.append(go.Scatter(y=data.loc[:, sensorDropdown[i]], name=name, yaxis=yaxis, opacity=1-0.2*i))
        
            

        if (labelButtonClicks != None and labelButtonClicks % 2 == 0):
            
            labelButtonTitle = 'Start Label'
            dragMode = 'zoom'
            # New label pressed
        
            if 'selections' in relayoutData.keys():

                x0 = relayoutData['selections'][0]['x0']
                x1 = relayoutData['selections'][0]['x1']
                x0 = round(x0)
                x1 = round(x1)

                if (x0 > x1):
                    temp = x0
                    x0 = x1
                    x1 = temp

                if (labelDropdown=='No Fault'):
                    color = 'green'
                    fault = 0
                elif (labelDropdown=='Fault 1'):
                    color = 'red'
                    fault = 1
                elif (labelDropdown=='Fault 2'):
                    color = 'orange'
                    fault  = 2
                elif (labelDropdown=='Fault 3'):
                    color = 'yellow'
                    fault = 3
                shapes.append({
                    'type':'rect',
                    'x0':x0,
                    'x1':x1,
                    'y0':0,
                    'y1':0.05,
                    'fillcolor':color,
                    'yref': 'paper',
                    'name':labelDropdown
                },)

                index_to_delete = next((index for index, shape in enumerate(shapes) if shape['name'] == 'highlight'), None)
                if index_to_delete is not None:
                    if (shapes[index_to_delete]['x0'] < x1 and shapes[index_to_delete]['x1'] > x0):
                        del shapes[index_to_delete]


                labels[x0:x1] = [fault] * (x1-x0)
                
            

                

        if (removeLabelClick == 1 and shapes!=[]):
            shapes.clear()
            labels[0:len(labels)] = [-1] * (len(labels))
            n = labels.count(-1)
            stat3 = 'Data points unlabelled: ' + str(n)
        if (undoLabelClick == 1 and shapes!=[]):
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
        
                index_to_delete = next((index for index, shape in enumerate(shapes) if shape['name'] == 'highlight'), None)
                if index_to_delete is not None:
                    del shapes[index_to_delete]
                

                if (int(currentPoint)==len(labels)):
                    # Create an alert to informt that there are no furher ponts
                    alert = True
                    alertMessage = 'You have reached the end of the data'
                else:
                    start = -1
                    end = -1
                    for i in range(int(currentPoint), len(labels)):
                        if(labels[i] == target):
                            start = i
                            for j in range(i, len(labels)):
                                if (labels[j] != target):
                                    end = j
                                    currentPoint = str(end)
                                    break
                            if (end == -1):
                                end = len(labels)  
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
                            'type':'rect',
                            'x0':start,
                            'x1':end,
                            'y0':0,
                            'y1':1,
                            'fillcolor':'grey',
                            'opacity':0.5,
                            'yref': 'paper',
                            'name': 'highlight',
                            'editable': False
                        },)
                    
                    
            
        if (findPrevClicked == 1):
            # if (target == 'Unlabelled Data Point'):
                 # Find the index of the shape to be deleted (by name)
                index_to_delete = next((index for index, shape in enumerate(shapes) if shape['name'] == 'highlight'), None)
                if index_to_delete is not None:
                        del shapes[index_to_delete]
                

                if (int(currentPoint)==0):
                    # Create an alert to informt that there are no furher ponts
                    alert = True
                    alertMessage = 'You have reached the start of the data'
                else:
                    start = -1
                    end = -1
                    
                    for i in range(int(currentPoint)-1, 0, -1):
                        if(labels[i] == target):
                            end = i
                            start = -1
                            for j in range(i, 0, -1):
                                if (labels[j] != target):
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
                            'type':'rect',
                            'x0':start,
                            'x1':end,
                            'y0':0,
                            'y1':1,
                            'fillcolor':'grey',
                            'opacity':0.5,
                            'yref': 'paper',
                            'name': 'highlight',
                            'editable': False
                        },)
            
        if (labelButtonClicks != None and labelButtonClicks % 2 == 1):
            labelButtonTitle = 'Confirm Label'
            dragMode = 'select'
        else: 
            labelButtonTitle = 'New Label'
            dragMode = 'pan'

        
        
        if (newAutoLabel == 1):
            shapes = [] 
            df = data.loc[:, sensorChecklist]
            print(df)

            if (clusterMethod == 'K Means'):
                labels = performKMeans(df,3)
            
            shapes = []
            x0 = 0
            x1 = x0
    
            # labels = [0, 0, 0, 1, 1, 1, 2, 3, 4]
            for i in range(1, len(labels)):
                
                if labels[i] != labels[i-1]:
                    
                    x1 = i
                    if (labels[x0]==0):
                        color = 'green'
                    elif (labels[x0]==1):
                        color = 'blue'
                    elif (labels[x0]==2):
                        color = 'orange'
                    elif (labels[x0]==3):
                        color = 'yellow'
                    elif (labels[x0]==4):
                        color = 'red'

                    shapes.append({
                        'type':'rect',
                        'x0':x0,
                        'x1':x1,
                        'y0':0,
                        'y1':0.05,
                        'fillcolor':color,
                        'yref': 'paper',
                    },)
                
                    x0 = i
            if (labels[x0]==0):
                color = 'green'
            elif (labels[x0]==1):
                color = 'blue'
            elif (labels[x0]==2):
                color = 'orange'
            elif (labels[x0]==3):
                color = 'yellow'
            elif (labels[x0]==4):
                color = 'red'
            shapes.append({
                        'type':'rect',
                        'x0':x0,
                        'x1':len(labels),
                        'y0':0,
                        'y1':0.05,
                        'fillcolor':color,
                        'yref': 'paper',
                    },)
            x_0 = 0
            x_1 = 20000
            
        
        
        layout = go.Layout(legend={'x':0, 'y':1.2}, xaxis=dict(range=[x_0, x_1]), dragmode=dragMode, yaxis=dict(fixedrange=True, title='Sensor Value', color='blue'), yaxis2=dict(fixedrange=True, overlaying='y', color='orange', side='right'), yaxis3=dict(fixedrange=True, overlaying='y', color='green',side='left', position=0.001,), yaxis4=dict( fixedrange=True, overlaying='y', color='red', side='right'),shapes=shapes)
    
        if t is not  None:
            selectData.append(go.Line(x=[t,t], y=[0,max], name='Selected Point', line=dict(color='black')  ))
            # print('plotted')

        fig = {'data': selectData,'layout': layout,}

        sensorDropdownStyle = {'display': 'block'}
        # if relayoutData:
        #     print(relayoutData)
        #     fig.update_layout(xaxis_range=relayoutData.get('xaxis.range'))

        
        

    if (switchViewButtonClicks % 2 == 1):
        #  3D SCATTER PLOT
        if (observeClusterClikcs % 2 == 0):

            # fig = px.scatter_3d(data, x='xmeas_1', y='xmeas_2', z='xmeas_3', text='Unnamed: 0', color_discrete_sequence=['black'])
            fig = px.scatter_3d(data, x=xAxis_dropdown_3D, y=yAxis_dropdown_3D, z=zAxis_dropdown_3D, color='Time', opacity=0.05)
            labelButtonTitle = 'New Label'
            print('T:')
            print(t)
            if t is not None:
                fig.add_scatter3d(x=[data[xAxis_dropdown_3D][t]],y=[data[yAxis_dropdown_3D][t]],z=[data[zAxis_dropdown_3D][t]], marker=dict(color='black'))

            sensorDropdownStyle = {'display': 'none'}
        elif (observeClusterClikcs % 2 == 1):
            labelButtonTitle = 'New Label'
            
            df = data
            df['labels'] = labels
            #  USE ^^ ELSEWHERE

            cluster_dataframes = []
            for label in range(-1, 3):
                # Filter data based on cluster label
                cluster_df = df[labels == label]
                cluster_dataframes.append(cluster_df)

            global colours

            print(cluster_dataframes)
            fig = px.scatter_3d()
            for label in range(0, 4):
                print(label)
                fig.add_trace(px.scatter_3d(cluster_dataframes[label], x=xAxis_dropdown_3D, y=yAxis_dropdown_3D, z=zAxis_dropdown_3D,
                                            color_discrete_sequence=colours[label]).data[0])


            sensorDropdownStyle = {'display': 'none'}

    # n = labels.count(-1)     
    # stat1 = 'Data points unlabelled: ' + str(n)
    # n = len(set(labels))
    # stat2 = 'No. Types Labels Placed: ', int(len(set(labels))-1)
    stat3 = 'No. Labels Placed: ', len(shapes)

    return fig, labelButtonTitle, 0, 0, 0, 0, stat1, stat2, stat3, alert, alertMessage, sensorDropdownStyle, 0


@app.callback(
    Output('shape-clicked', 'children'),
    [Input(mainGraph, 'clickData')]
)
def update_textbox(click_data):
    if click_data is None:
        return "Click on a shape to see its information"
    else:
        # Find the shape that was clicked
        clicked_shape_info = None
        print(click_data)
        for shape in shapes:
            if (
                shape['x0'] <= click_data['points'][0]['x'] and
                shape['x1'] >= click_data['points'][0]['x']
            ):
                print('got here')
                clicked_shape_info = shape['name']
                print(clicked_shape_info)
                break

        if clicked_shape_info:
            return f"Clicked Shape Info: {clicked_shape_info}"
        else:
            return "No shape clicked"



@app.callback(
    Output('points-output', 'children'),
    [Input(mainGraph, 'clickData')],
    
    
)
def display_coordinates(click_data):
    
    if click_data is not None and 'points' in click_data:
        point = click_data['points'][0]
        x, y = point['x'], point['y']
        global t 
        t = x
        
        return f'Time = {x}'
        
    else:
        return 'Click on a point to display its coordinates',
    


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
        return data.columns, 0, 0
    else:
        return [], 0, 0
    

    

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


