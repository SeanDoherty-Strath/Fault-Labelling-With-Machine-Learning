import dash
from dash import dcc, html, Input, Output
import pandas as pd

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
        # Allow multiple files to be uploaded
        multiple=True
    ),
    html.Div(id='output-data-upload')
])

# Define callback to parse the contents of the uploaded CSV file
@app.callback(Output('output-data-upload', 'children'),
              [Input('upload-data', 'contents')],
              [dash.dependencies.State('upload-data', 'filename')])
def update_output(contents, filenames):
    if contents is not None:
        # Parse the CSV file
        df = parse_contents(contents, filenames)
        # Display the CSV content as a DataTable
        return html.Div([
            html.H5(f'Uploaded File: {filenames}'),
            html.Hr(),
            html.Div([
                dcc.DataTable(
                    data=df.to_dict('rows'),
                    columns=[{'name': i, 'id': i} for i in df.columns],
                    style_table={'overflowX': 'scroll'},
                )
            ])
        ])

def parse_contents(contents, filenames):
    content_type, content_string = contents[0].split(',')
    decoded = base64.b64decode(content_string)
    # Assume that the user uploaded a CSV file
    df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    return df

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
