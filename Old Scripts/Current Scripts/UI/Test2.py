import dash
from dash import dcc, html, Input, Output
import plotly.graph_objs as go

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(
        id='3d-scatter-plot',
        config={'editable': True},
        figure={
            'data': [
                go.Scatter(
                    x=[1, 2, 3, 4, 5],
                    y=[5, 4, 3, 2, 1],
                    mode='markers',
                    marker={
                        'size': 10,
                        'opacity': 0.8,
                    },
                )
            ],
            'layout': go.Layout(
                margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
                dragmode='lasso'
            )
        }
    ),
    html.Div(id='selected-points')
])


@app.callback(
    Output('selected-points', 'children'),
    [Input('3d-scatter-plot', 'selectedData')]
)
def display_selected_data(selectedData):
    if selectedData is None:
        return "No points selected"

    selected_points = selectedData['points']
    print(selectedData)
    if not selected_points:
        return "No points selected"

    output_text = "Selected points:\n\n"
    for point in selected_points:
        output_text += f"X: {point['x']}, Y: {point['y']}\n"

    return dcc.Markdown(output_text)


if __name__ == '__main__':
    app.run_server(debug=True)
