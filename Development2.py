import dash
from dash import Output, Input, dcc, html
import plotly.express as px

app = dash.Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='graph'),
    ], style={'display': 'inline-block', 'width': '50%'}),

    html.Div([
        dcc.Dropdown(
            id='dropdown-component-1',
            options=[
                {'label': 'Option 1', 'value': 'option1'},
                {'label': 'Option 2', 'value': 'option2'}
            ],
            value='option1',
            style={'margin': '10px', 'text-align': 'center'}

        ),
        dcc.Dropdown(
            id='dropdown-component-2',
            options=[
                {'label': 'Option A', 'value': 'optionA'},
                {'label': 'Option B', 'value': 'optionB'}
            ],
            value='optionA',
            style={'margin': '10px', 'text-align': 'center'}

        ),
        dcc.Dropdown(
            id='dropdown-component-3',
            options=[
                {'label': 'Choice X', 'value': 'choiceX'},
                {'label': 'Choice Y', 'value': 'choiceY'}
            ],
            value='choiceX',
            style={'margin': '10px', 'text-align': 'center'}

        ),
    ], style={'display': 'inline-block', 'width': '50%', 'vertical-align': 'top'})
])


@app.callback(
    Output('graph', 'figure'),
    Input('dropdown-component-1', 'value'),
    Input('dropdown-component-2', 'value'),
    Input('dropdown-component-3', 'value')
)
def update_graph(selected_option1, selected_option2, selected_option3):
    # Customize this part to update the graph based on the selected options
    title = f'Graph for {selected_option1} - {selected_option2} - {selected_option3}'
    fig = px.scatter(x=[1, 2, 3, 4], y=[10, 11, 12, 13], title=title)
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
