import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
from dash import dcc
import pandas as pd

# Sample data
data = {
    'timestamp': ['2024-03-14 10:00:00', '2024-03-14 10:15:00', '2024-03-14 10:30:00'],
    'user': ['User1', 'User2', 'User3'],
    'comment': ['This is comment 1', 'This is comment 2', 'This is comment 3']
}

# Create DataFrame
comments = pd.DataFrame(data)

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the modal
modal = html.Div(
    [
        dbc.Modal(
            children=[
                dbc.ModalHeader("Comments"),
                # html.Div("This is the content of the modal", children=[
                html.Div(style={'flex-direction': 'row', 'display': 'flex', 'justify-content': 'space-evenly'},  children=[
                    dcc.Markdown("Timestamp"),
                    dcc.Markdown("<user>"),
                    dcc.Markdown("Comment"),
                ]),
                html.Div(style={'flex-direction': 'row', 'display': 'flex', 'justify-content': 'space-evenly'},  children=[
                    dcc.Markdown("Timestamp"),
                    dcc.Markdown("<user>"),
                    dcc.Markdown("Comment"),
                ]),
                html.Div(style={'flex-direction': 'row', 'display': 'flex', 'justify-content': 'space-evenly'},  children=[
                    dcc.Markdown("Timestamp"),
                    dcc.Markdown("<user>"),
                    dcc.Markdown("Comment"),
                ]),

                dbc.ModalFooter(children=[
                    dcc.Input(id='commentInput',
                              type='text', value='Comment'),
                    dcc.Input(id='usernameInput', type='text', value='Name'),
                    html.Button(id='addComment')]
                ),
            ],
            id="commentModal",
            centered=True,
            size="md",
        ),
        html.Button("Open Modal", id="open-modal"),



    ]
)

# Layout of the app
app.layout = html.Div([
    modal
])

# Callback to control the modal


@app.callback(
    Output("commentModal", "is_open"),
    Output("commentModal", 'children'),
    Output("addComment", "n_clicks"),
    Input("open-modal", "n_clicks"),
    Input("addComment", "n_clicks"),
    # Input("close-modal", "n_clicks"),
    State("commentModal", "is_open"),
    State("commentInput", 'value'),
    State("usernameInput", 'value')
)
def toggle_modal(open_clicks, is_open, addComment,  commentInput, usernameInput):

    global comments
    if (addComment):
        comments = comments._append({'timestamp': '2024-03-14 10:00:00',
                                     'user': usernameInput, 'comment': commentInput}, ignore_index=True)

    modalChidren = [dbc.ModalHeader("Comments")]

    for i in range(comments.shape[0]):
        # print(comments.iloc[0, i])

        modalChidren.append(
            html.Div(style={'flex-direction': 'row', 'display': 'flex', 'justify-content': 'space-evenly'},  children=[
                dcc.Markdown(comments.iloc[i, 0]),
                dcc.Markdown(comments.iloc[i, 1]),
                dcc.Markdown(comments.iloc[i, 2]),
            ]),)

    modalChidren.append(dbc.ModalFooter(
        dbc.ModalFooter(children=[
            dcc.Input(id='commentInput', type='text', value='Comment'),
            dcc.Input(id='usernameInput', type='text', value='Name'),
            html.Button(id='addComment')]
        ),
    ),)
    if (addComment):
        return True,  modalChidren, 0

    if open_clicks:
        return not is_open, modalChidren, 0
    return is_open, modalChidren, 0


if __name__ == "__main__":
    app.run_server(debug=True)
