import dash_core_components as dcc
import dash
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from dash import html 
import dash_bootstrap_components as dbc

data = pd.read_csv("Data/UpdatedData.csv")
data = data.iloc[:, 4:]

fig = px.line()
fig.update_layout(dragmode=False)
figure1_graph = dcc.Graph(figure=fig)

figure2_dropdown = dcc.Checklist(options=data.columns, style={'fontsize':18, 'fontFamily':'Comic Sans MS'})
figure2_header = dcc.Markdown(children='Sensors', style={'fontsize':30, 'fontWeight':'bold', 'textAlign':'center', 'fontFamily':'Comic Sans MS'})


figure3_label = dcc.Dropdown(value='No Fault', options=['Fault 1', 'Fault 2', 'Fault 3', 'No Fault'])

slider = dcc.Slider(0, 10, 1, value=0)
pan = dcc.Slider(0, 2000, 100, value=0)


xAxis_dropdown_3D = dcc.Dropdown(value=data.columns[0], options = data.columns)
yAxis_dropdown_3D = dcc.Dropdown(value=data.columns[1], options = data.columns)
zAxis_dropdown_3D = dcc.Dropdown(value=data.columns[2], options = data.columns)



fourGraphs =  html.Div(style={'margin': '0', 'padding': '0'}, children=[
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
                'layout':go.Layout(height=350)},
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
                'layout':go.Layout(height=350)},
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
                'layout':go.Layout(height=350)},
                config={'displayModeBar': False}
        ),
])])

