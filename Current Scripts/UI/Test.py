import dash
import dash_bootstrap_components as dbc
from dash import dcc, html
from dash.dependencies import Input, Output, State

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

app.layout = html.Div([
    html.Button("Open Modal", id="open-modal-button"),
    
    dbc.Modal(
        [
            dbc.ModalHeader("Example Modal Header"),
            dbc.ModalBody("This is the content of the modal"),
            dbc.ModalFooter(
                dbc.Button("Close", id="close-modal-button", className="ml-auto")
            ),
        ],
        id="example-modal",
    ),
])

@app.callback(
    Output("example-modal", "is_open"),
    [Input("open-modal-button", "n_clicks"), Input("close-modal-button", "n_clicks")],
    [State("example-modal", "is_open")],
)
def toggle_modal(n1, n2, is_open):
    if n1 or n2:
        return not is_open
    return is_open

if __name__ == '__main__':
    app.run_server(debug=True)
