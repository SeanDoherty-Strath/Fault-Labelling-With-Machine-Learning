import dash
from dash import html, dcc, Input, Output
import pandas as pd
import numpy as np

# Sample data for the graph
np.random.seed(42)
x = np.arange(100)
y = np.random.randn(100)

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='scrollable-graph',
    ),
    dcc.RangeSlider(
        id='x-range-slider',
        min=0,
        max=len(x) - 1,
        step=1,
        marks={i: str(i) for i in range(len(x))},
        value=[0, len(x) - 1],
    ),
])


@app.callback(
    Output('scrollable-graph', 'figure'),
    Input('x-range-slider', 'value')
)
def update_x_range(selected_range):
    start, end = selected_range
    x_range = x[start:end+1]
    y_range = y[start:end+1]

    # Create a figure with the updated x-axis range
    fig = {
        'data': [{'x': x_range, 'y': y_range, 'type': 'scatter'}],
        'layout': {
            'xaxis': {'range': [x[start], x[end]]},
            'yaxis': {'title': 'Value'},
            'title': 'Horizontal Scrollable Graph'
        }
    }

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
