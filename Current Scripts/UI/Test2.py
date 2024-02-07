import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

# Example data with x, y, z, and time values
data = {'x': [1, 2, 3, 4],
        'y': [10, 15, 13, 17],
        'z': [5, 8, 11, 14],
        'time': ['2024-01-30 12:00:00', '2024-01-30 12:15:00', '2024-01-30 12:30:00', '2024-01-30 12:45:00']}

scatter_trace = go.Scatter3d(
    x=data['x'],
    y=data['y'],
    z=data['z'],
    mode='markers',
    text=data['time'],  # Set time values as text labels
    marker=dict(
        size=10,
        color='rgb(0, 0, 255)',
        opacity=0.8
    )
)

app.layout = html.Div([
    dcc.Graph(
        id='scatter-plot',
        figure={
            'data': [scatter_trace],
            'layout': go.Layout(
                title='3D Scatter Plot with Time',
                scene=dict(
                    xaxis=dict(title='X Axis'),
                    yaxis=dict(title='Y Axis'),
                    zaxis=dict(title='Z Axis')
                )
            )
        }
    ),
    html.Div(id='hover-info')
])

@app.callback(
    Output('hover-info', 'children'),
    [Input('scatter-plot', 'hoverData')]
)
def display_hover_info(hover_data):
    if hover_data is not None and 'points' in hover_data:
        point = hover_data['points'][0]
        time_value = point['text']
        return f'Hovered Point Time: {time_value}'
    else:
        return 'Hover over a point to display its time'

if __name__ == '__main__':
    app.run_server(debug=True)
