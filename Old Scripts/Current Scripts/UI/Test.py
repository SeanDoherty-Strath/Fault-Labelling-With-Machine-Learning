import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import io
import base64  # Import base64 module

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    dcc.Upload(
        id='upload-data',
        children=html.Div([
            'Drag and Drop or ',
            html.A('Select Files')
        ]),
        style={
            'width': '100%',
            'height': '60px',
            'lineHeight': '60px',
            'borderWidth': '1px',
            'borderStyle': 'dashed',
            'borderRadius': '5px',
            'textAlign': 'center',
            'margin': '10px'
        },
        multiple=False
    ),
    html.Div(id='output-data-upload')
])

# Define callback to parse CSV and update dataframe


@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')])
def update_output(contents):
    if contents is not None:
        content_type, content_string = contents.split(',')
        decoded = io.StringIO(base64.b64decode(content_string).decode('utf-8'))
        df = pd.read_csv(decoded)
        return html.Div([
            html.H5('Uploaded File Content:'),
            dcc.Textarea(
                value=df.to_string(),
                readOnly=True,
                style={'width': '100%', 'height': '300px'}
            ),
            html.Hr(),
            html.H5('Dataframe Info:'),
            html.Pre(df.info())
        ])


# Run the Dash app
if __name__ == '__main__':
    app.run_server(debug=True)
