import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import json

app = dash.Dash(__name__)

# Initialize empty selected box
selected_box = None

app.layout = html.Div([
    dcc.Graph(
        id='plot-container',
        config={'scrollZoom': False},  # Disable zooming
        figure=make_subplots(rows=1, cols=1)
    ),
    dcc.Store(id='selected-box', data=selected_box)
])


@app.callback(
    Output('plot-container', 'figure'),
    Input('selected-box', 'data')
)
def update_plot(selected_data):
    # Create a figure with a scatter plot
    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(x=[1, 2, 3, 4], y=[
                  10, 11, 12, 13], mode='lines', name='Plot'))
    fig.update_layout(title='Plot', xaxis_title='X-Axis', yaxis_title='Y-Axis')

    # If a box is selected, color the selected section green
    if selected_data:
        x0, y0, x1, y1 = selected_data
        fig.add_shape(
            type='rect',
            x0=x0, y0=y0, x1=x1, y1=y1,
            line=dict(color='green', width=2),
            fillcolor='green',
            opacity=0.3
        )

    return fig


@app.callback(
    Output('selected-box', 'data'),
    Input('plot-container', 'relayoutData')
)
def select_box(relayoutData):
    # Retrieve the coordinates of the selected box
    if 'xaxis.range' in relayoutData:
        x0, x1 = relayoutData['xaxis.range']
        y0, y1 = relayoutData['yaxis.range']
        return x0, y0, x1, y1
    return None


if __name__ == '__main__':
    app.run_server(debug=True)
