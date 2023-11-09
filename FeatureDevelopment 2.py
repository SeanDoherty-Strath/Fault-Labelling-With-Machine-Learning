import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd


# Create a sample DataFrame with multiple scatter plots
data = pd.DataFrame({
    'x': [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    'y': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
    'group': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'B', 'A', 'B'],
})

# Create a Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='my-plot'),
])


@app.callback(
    Output('my-plot', 'figure'),
    Input('my-plot', 'relayoutData')
)
def update_plot(relayoutData):
    # Create a scatter plot for each group
    fig = px.scatter(data, x='x', y='y', color='group',
                     symbol='group', width=600, height=400)

    # Customize the layout
    fig.update_layout(
        title="Multiple Scatters on the Same Point",
        xaxis=dict(title='X-axis'),
        yaxis=dict(title='Y-axis'),
    )

    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
