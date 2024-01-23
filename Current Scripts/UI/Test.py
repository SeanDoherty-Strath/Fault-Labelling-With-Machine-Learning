import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd

# Sample data
df = pd.DataFrame({
    'X': [1, 2, 3, 4, 5],
    'Y1': [10, 11, 12, 13, 14],
    'Y2': [15, 14, 13, 12, 11],
    'Y3': [5, 4, 3, 2, 1],
    'Y4': [8, 7, 6, 5, 4]
})

# Initialize the Dash app
app = dash.Dash(__name__)

# Define layout
app.layout = html.Div(children=[
    html.H1("Graphs with Four Subplots"),

    # First Row
    html.Div([
        dcc.Graph(
            id='subplot-graph-1',
            figure=px.line(df, x='X', y='Y1', title='Subplot 1').update_layout(showlegend=False)
        ),
        dcc.Graph(
            id='subplot-graph-2',
            figure=px.line(df, x='X', y='Y2', title='Subplot 2').update_layout(showlegend=False)
        ),
    ], className='row'),

    # Second Row
    html.Div([
        dcc.Graph(
            id='subplot-graph-3',
            figure=px.line(df, x='X', y='Y3', title='Subplot 3').update_layout(showlegend=False)
        ),
        dcc.Graph(
            id='subplot-graph-4',
            figure=px.line(df, x='X', y='Y4', title='Subplot 4').update_layout(showlegend=False)
        ),
    ], className='row')
])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
 