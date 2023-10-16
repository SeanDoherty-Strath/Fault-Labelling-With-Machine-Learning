from dash import Dash, dcc, Output, Input
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px


# COMPONENTS
df = pd.read_csv(
    'C:/Users/sdohe/Documents/4th Year Project/Scripts/TestData3.csv')
# df = pd.read_csv("C:/Users/sdohe/OneDrive/Documents/HPS_App/PinLocation/Scripts/TestData2.csv")

df2 = px.data.medals_long()
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
mytext = dcc.Markdown(
    children="# My Graph",
)


myGraph = dcc.Graph(figure={})
dropdown = dcc.Dropdown(
    options=["Data 1", "Data 2", "Data 3", "Data 1 + 2",
             "Data 1 + 2 (scatter)", "Data 1, 2, + 3"],
    value="Data 1",  # initial value displayed when page first loads
    clearable=False,
)


# LAYOUT
app.layout = dbc.Container([mytext, myGraph, dropdown])


@app.callback(
    [Output(myGraph, component_property="figure")],
    Input(dropdown, component_property="value"),
)
# function arguments come from the component property of the Input
def update_graph(user_input):
    if user_input == "Data 1":
        fig = px.line(df, x="Date", y="Sensor 1",)

    elif user_input == "Data 2":
        fig = px.line(df, x="Date", y="Sensor 2")

    elif user_input == "Data 3":
        fig = px.line(df, x="Date", y="Sensor 3")

    elif user_input == "Data 1 + 2":
        fig = px.line(df, x="Date", y=["Sensor 1", "Sensor 2"])

    elif user_input == "Data 1 + 2 (scatter)":
        fig = px.scatter(df, x="Sensor 1", y="Sensor 2")

    elif user_input == "Data 1, 2, + 3":
        fig = px.scatter_3d(df, x="Sensor 1", y="Sensor 2",
                            z="Sensor 3")

    # returned objects are assigned to the component property of the Output
    return fig,


@app.callback(
    Output(mytext, component_property='children'),
    Input(myGraph, component_property='clickData')
)
def display_selected_time(clickData):
    if clickData is None:
        return "Click on the graph to display the time."

    selected_time = clickData['points'][0]['x']

    return "# Selected Time: " + selected_time,


# Run app
if __name__ == "__main__":
    app.run_server(port=8053)
