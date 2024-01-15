import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

# Define the layout
app.layout = html.Div(style={'background': 'linear-gradient(to bottom, blue, #000000)',
                             'height': '100vh', 'display': 'flex', 'justify-content': 'center', 'flex-direction': 'column', 'align-items': 'center'},
                      children=[
                        #   html.H1("Fault Label", style={'color': 'white', 'font-size': '2.5em', 'margin-left': '20px'}),


                          html.Div(style={'width': '90%', 'height': '45%', 'background-color': 'white',
                                        'margin':'20px', 'border-radius': '10px', 'padding': '20px'},
                                   children=[
                                       dcc.Graph(id='box1-graph', config={'displayModeBar': False},style={'width': '100%', 'height': '100%'}
)
                                   ]
                                   ),
                       html.Div(style={'width': '90%', 'height': '45%',  'display': 'flex', 'justify-content': 'space-between', 'align-items': 'center',
                                    'margin':'20px', },
                                    children=[
                                        html.Div(style={'border-radius': '10px', 'width': '32%', 'height': '100%', 'background-color': 'white'}),
                                        html.Div(style={'border-radius': '10px','width': '32%', 'height': '100%', 'background-color': 'white'}),
                                        html.Div(style={'border-radius': '10px','width': '32%', 'height': '100%', 'background-color': 'white'}),
                                    ]
        
                        ),
                      ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
