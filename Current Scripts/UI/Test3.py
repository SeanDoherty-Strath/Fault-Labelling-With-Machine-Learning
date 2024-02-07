import dash
from dash import dcc, html
from dash.dependencies import Input, Output

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [
                {'x': [1, 2, 3, 4], 'y': [10, 15, 13, 17], 'type': 'scatter', 'mode': 'markers'},
            ],
            'layout': {
                'title': 'Clickable Scatter Plot'
            }
        }
    ),
    html.Div(id='point-coordinates-output')
])

@app.callback(
    Output('point-coordinates-output', 'children'),
    [Input('scatter-plot', 'clickData')]
)
def display_coordinates(click_data):
    if click_data is not None and 'points' in click_data:
        point = click_data['points'][0]
        x, y = point['x'], point['y']
        return f'Clicked Point Coordinates: x={x}, y={y}'
    else:
        return 'Click on a point to display its coordinates'

if __name__ == '__main__':
    app.run_server(debug=True)
