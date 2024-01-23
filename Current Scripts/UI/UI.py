import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
from Components import figure1_graph, figure2_dropdown, figure2_header, figure3_label, slider, pan, fourGraphs, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D
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

shapes = []

# Define the layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, blue, #000000)', 'height': '100vh', 'display': 'flex', 'justify-content': 'center', 'flex-direction': 'column', 'align-items': 'center'},
             children=[
                      # Top Graph
                 html.Div(style={'height':'5%', 'width':'90%'}, children=[
                    html.H1(children='Fault Labeller', style={'fontsize':100, 'color':'white', 'fontWeight':'bold', 'textAlign':'left', 'fontFamily':''})
                 ]
                 ),
                    html.Div(style={'overflow':'scroll', 'width':'90%', 'height':'55%', 'margin':'20px', 'border-radius': '10px', 'padding': '20px', 'background-color': 'white'},
                      children=[
                          dcc.Markdown(id='action', children='Latest Action: ', style={'color': 'black'}),
                          html.Button('Switch View', id='switchView'),
                          
                          
                          figure1_graph, fourGraphs
                      ]
                      ),
                      # Bottom 3 boxes
                      html.Div(style={'width': '90%', 'height': '35%',  'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center', 'margin':'20px', },
                                    children=[
                                        html.Div(style={'overflow':'scroll', 'border-radius': '10px', 'width': '24%', 'height': '100%',  'background-color': 'white'}, children=[figure2_header, figure2_dropdown, xAxis_dropdown_3D, yAxis_dropdown_3D, zAxis_dropdown_3D]),
                                        html.Div(style={'width': '24%', 'height': '100%', }, children=[
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'}),
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '6%', }),
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'})
                                        ]),
                                        html.Div(style={'border-radius': '10px','width': '24%', 'height': '100%', 'background-color': 'white'}, children=[ figure3_label, html.Button('Confirm Label', id='newLabel'), html.Button('Remove Label', id='removeLabels'), html.Button('Undo Label', id='undoLabel')]),
                                        html.Div(style={'width': '24%', 'height': '100%', }, children=[    
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'}, 
                                                    children=[slider,pan]
                                                    ),
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '6%', }),
                                            html.Div(style={'border-radius': '10px','width': '100%', 'height': '47%', 'background-color': 'white'})
                                        ]),
                                    ]
                        ),
                      ])



@app.callback(
    Output(figure2_dropdown, 'value'),
    Input(figure2_dropdown, 'value')
)
def updatedText(values):
    if (values == None):
        return ['xmeas_1']
    if (len(values) > 4):
        values = values[1:]

    return  values


@app.callback(
    Output(figure1_graph, "figure"),
    Output(figure2_dropdown, 'style'),
    Output(xAxis_dropdown_3D, 'style'),
    Output(yAxis_dropdown_3D, 'style'),
    Output(zAxis_dropdown_3D, 'style'),
    Output(fourGraphs, 'style'),
    Output(figure1_graph, 'style'),
    Input(figure2_dropdown, 'value'),
    Input('newLabel', 'n_clicks'),
    Input(slider, 'value'),
    Input(pan, 'value'),
    Input('switchView', 'n_clicks')
)
def updatedGraph(values, newLabelButton, zoom, pan, switchViewClicks):
    if (switchViewClicks == None):
        switchViewClicks = 0
    if (switchViewClicks % 3 == 0):

        # TIME BASED VIEW
        figure2_dropdown_style = {"display": "block"}
        xAxis_dropdown_3D_style = {"display": "none"}
        yAxis_dropdown_3D_style = {"display": "none"}
        zAxis_dropdown_3D_style = {"display": "none"}
        fourGraphs_style = {"display":"none"}
        figure1_graph_style = {"display":"block"}

        selectData = []
        for i in range(len(values)):
            name = values[i]
            yx = 'y' + str(i+1)
            selectData.append(go.Scatter(y=data.loc[:, values[i]], name=name, yaxis=yx))
        
        x0 = pan
        zoom = 10-zoom
        x1 = x0+zoom*200+1
        layout = go.Layout(xaxis=dict(range=[x0, x1]), dragmode='select', legend={'x':0, 'y':1.2}, yaxis=dict(title='Sensor Value', color='blue'), yaxis2=dict(overlaying='y', color='orange', side='right'), yaxis3=dict(overlaying='y', color='green',side='left', position=0.001,), yaxis4=dict( overlaying='y', color='red', side='right'),
                        shapes=shapes)
        
        fig = {'data': selectData,
        'layout': layout,
        }

    elif (switchViewClicks % 3 == 1):

        # 4 time based views (or more)
        
        
        figure2_dropdown_style = {"display": "none"}
        xAxis_dropdown_3D_style = {"display": "none"}
        yAxis_dropdown_3D_style = {"display": "none"}
        zAxis_dropdown_3D_style = {"display": "none"}
        fourGraphs_style = {"display":"block"}
        figure1_graph_style = {"display":"none"}
        fig = px.line()
        
    elif (switchViewClicks % 3 == 2):
        # 3D View
        fig = performKMeans(data)
        figure2_dropdown_style = {"display": "none"}
        xAxis_dropdown_3D_style = {"display": "block"}
        yAxis_dropdown_3D_style = {"display": "block"}
        zAxis_dropdown_3D_style = {"display": "block"}
        fourGraphs_style = {"display":"none"}
        figure1_graph_style = {"display":"block"}

    return fig, figure2_dropdown_style, xAxis_dropdown_3D_style, yAxis_dropdown_3D_style, zAxis_dropdown_3D_style, fourGraphs_style, figure1_graph_style


@app.callback(
    Output('action', "children"),
    Output('newLabel', 'n_clicks'),
    Output('removeLabels', 'n_clicks'),
    Output('undoLabel', 'n_clicks'),
    Input('newLabel', 'n_clicks'),
    Input('removeLabels', 'n_clicks'),
    Input('undoLabel', 'n_clicks'),
    State(figure1_graph, 'relayoutData'),
    State(figure3_label, 'value'),
)
def drawRectangle(newLabel, removeLabel, undoLabel, relayoutData, label):
    actionmessage = ''
    print(newLabel)
    print(removeLabel)
    
    if (newLabel == 1):
    
        # New label pressed
        if (relayoutData == None):
            return 'Latest Action:'
        if 'selections' not in relayoutData.keys():
            return 'Latest Action:'
        
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

        shapes.append({
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
        shapes.clear()
        actionmessage = 'Latest Action: Labels Removed'

    elif (undoLabel == 1):
        shapes.pop()
        actionmessage = 'Latest Action: Labels Undone'

    return actionmessage,0, 0, 0









# @app.callback(
#     Output(figure1_graph, "figure"),
#     Input(figure2_dropdown, 'value')
# )
# def updateGraph(value):
#   fig = px.line(data, y=value)
#   return fig
  
        
    


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
