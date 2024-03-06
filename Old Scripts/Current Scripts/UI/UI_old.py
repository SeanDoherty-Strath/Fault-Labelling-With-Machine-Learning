import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from Components import mainGraph, zoom, pan, fourGraphs, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D, stats, exportCSV, faultFinder
from Components import title, sensorDropdown, sensorHeader, labelDropdown
from myFunctions import changeText, updateGraph, performKMeans
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from sklearn.cluster import KMeans
import pandas as pd


app = dash.Dash(__name__)

# DATA
data = pd.read_csv("Data/UpdatedData.csv")
data = data.iloc[:, 4:]

labels = []

# Define the layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, blue, #000000)', 'height': '100vh', 'display': 'flex', 'justify-content': 'center', 'flex-direction': 'column', 'align-items': 'center'},children=[
                html.Div(style={'height':'5%', 'width':'90%'}, 
                        children=[title]),
                    
                # Top Box
                html.Div(style={'overflow':'scroll', 'width':'90%', 'height':'55%', 'margin':'20px', 'border-radius': '10px', 'padding': '20px', 'background-color': 'white'},
                    children=[
                        dcc.Markdown(id='latestAction', children='Latest Action: '),
                        html.Button('Switch View', id='switchView'),
                        mainGraph, 
                        fourGraphs]),

                # Bottom boxes
                html.Div(style={'width': '90%', 'height': '35%',  'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin':'20px', },
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
                                    zoom, 
                                    pan]), 

                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '6%', }),
                            
                            #  Box 3
                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'}, 
                                children=[
                                    faultFinder])]),
                            
                            # Box 4
                            html.Div(style={'border-radius': '10px','width': '24%', 'height': '100%', 'background-color': 'white'}, 
                                     children=[ 
                                         labelDropdown, 
                                         html.Button('Confirm Label', id='newLabel'), 
                                         html.Button('Remove Label', id='removeLabels'), 
                                         html.Button('Undo Label', id='undoLabel'),
                                         html.Button('Start Label', id='startLabel')]),

                            # Box 5
                            html.Div(style={'width': '24%', 'height': '100%', }, 
                                     children=[    
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'}, 
                                                    children=[stats]
                                                    ),
                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '6%', }),
                            
                            # Box6
                            html.Div(style={'border-radius': '10px', 'width': '100%', 'height': '47%', 'background-color': 'white', 'display': 'flex','justify-content': 'center',}, 
                                children=[
                                    exportCSV]
        )]),]),])



@app.callback(
    Output(sensorDropdown, 'value'),
    Input(sensorDropdown, 'value')
)
def updatedText(values):
    if (values == None):
        return []
    if (len(values) > 4):
        values = values[1:]

    return  values


@app.callback(
    # Graphs
    Output(mainGraph, "figure"),
    Output(mainGraph, 'style'),
    Output(fourGraphs, 'style'),

    # hide / show dropdowns
    Output(sensorDropdown, 'style'),
    Output(xAxis_dropdown_3D, 'style'),
    Output(yAxis_dropdown_3D, 'style'),
    Output(zAxis_dropdown_3D, 'style'),

    Input(sensorDropdown, 'value'),
    Input('startLabel', 'n_clicks'),
    Input('newLabel', 'n_clicks'),
    Input(zoom, 'value'),
    Input(pan, 'value'),
    Input('switchView', 'n_clicks')
)
def updatedGraph(values, startLabelClicks, newLabelClicks, zoom, pan, switchViewClicks):
    

    
    if (switchViewClicks == None):
        switchViewClicks = 0
    if (switchViewClicks % 3 == 0):

        # TIME BASED VIEW
        sensor_dropdown_style = {"display": "block"}
        xAxis_dropdown_3D_style = {"display": "none"}
        yAxis_dropdown_3D_style = {"display": "none"}
        zAxis_dropdown_3D_style = {"display": "none"}
        fourGraphs_style = {"display":"none"}
        mainGraph_style = {"display":"block"}

        selectData = []
        for i in range(len(values)):
            name = values[i]
            yx = 'y' + str(i+1)
            selectData.append(go.Scatter(y=data.loc[:, values[i]], name=name, yaxis=yx))
        
        x0 = pan
        zoom = 10-zoom
        x1 = x0+zoom*200+1
        layout = go.Layout(xaxis=dict(range=[x0, x1]), dragmode='zoom', legend={'x':0, 'y':1.2}, yaxis=dict(title='Sensor Value', color='blue'), yaxis2=dict(overlaying='y', color='orange', side='right'), yaxis3=dict(overlaying='y', color='green',side='left', position=0.001,), yaxis4=dict( overlaying='y', color='red', side='right'),
                        shapes=labels)
        
        fig = {'data': selectData,'layout': layout,}

    elif (switchViewClicks % 3 == 1):

        # 4 time based views (or more)
        sensor_dropdown_style = {"display": "none"}
        xAxis_dropdown_3D_style = {"display": "none"}
        yAxis_dropdown_3D_style = {"display": "none"}
        zAxis_dropdown_3D_style = {"display": "none"}
        fourGraphs_style = {"display":"block"}
        mainGraph_style = {"display":"none"}
        fig = px.line()
        
    elif (switchViewClicks % 3 == 2):
        # 3D View
        fig = performKMeans(data)
        sensor_dropdown_style = {"display": "none"}
        xAxis_dropdown_3D_style = {"display": "block"}
        yAxis_dropdown_3D_style = {"display": "block"}
        zAxis_dropdown_3D_style = {"display": "block"}
        fourGraphs_style = {"display":"none"}
        mainGraph_style = {"display":"block"}

    

    return fig, mainGraph_style, fourGraphs_style, sensor_dropdown_style, xAxis_dropdown_3D_style, yAxis_dropdown_3D_style, zAxis_dropdown_3D_style, 




@app.callback(
    Output('latestAction', "children"),
    Output('newLabel', 'n_clicks'),
    Output('removeLabels', 'n_clicks'),
    Output('undoLabel', 'n_clicks'),

    Input('newLabel', 'n_clicks'),
    Input('removeLabels', 'n_clicks'),
    Input('undoLabel', 'n_clicks'),

    State(mainGraph, 'relayoutData'),
    State(labelDropdown, 'value'),
)
def drawRectangle(newLabel, removeLabel, undoLabel, relayoutData, label):
    actionmessage = ''
    print(newLabel)
    print(removeLabel)
    
    if (newLabel == 1):
    
        # New label pressed
        if (relayoutData == None):
            return 'Latest Action:', 0, 0, 0
        if 'selections' not in relayoutData.keys():
            return 'Latest Action:', 0, 0, 0
        
        x0 = relayoutData['selections'][0]['x0']
        x1 = relayoutData['selections'][0]['x1']

        if (label=='No Fault'):
            color = 'green'
        elif (label=='Fault 1'):
            color = 'red'
        elif (label=='Fault 2'):
            color = 'orange'
        else:
            color = 'yellow'

        labels.append({
            'type':'rect',
            'x0':x0,
            'x1':x1,
            'y0':0,
            'y1':0.05,
            'fillcolor':color,
            'yref': 'paper',
        },)
        actionmessage = 'Latest Action: ' + str(round(x0)) + ' to ' + str(round(x1)) + ' labelled as ' + label
    elif (removeLabel == 1):
        labels.clear()
        actionmessage = 'Latest Action: Labels Removed'

    elif (undoLabel == 1):
        labels.pop()
        actionmessage = 'Latest Action: Labels Undone'

    return actionmessage,0, 0, 0








    


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
