import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
import keras
from keras import layers
import pandas as pd
import pyreadr
import math
from dash import Dash, dcc, Output, Input, html, State
import dash_bootstrap_components as dbc
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go


# A function that normalizes data
def normalize_dataframe(df):
    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


fig = px.line({})
myGraph = dcc.Graph(figure=fig)

xAxis = dcc.Dropdown(options=[0, 1, 2], value=0)


# LAYOUT
app = Dash(__name__, external_stylesheets=[
           dbc.themes.SOLAR])  # always the same
app.layout = dbc.Container(
    [
        myGraph,
        xAxis
    ]
)


@app.callback(
    Output(myGraph, "figure"),
    Input(xAxis, "value"),
)
def reduceDataSet(xAxis):
    print('Started Process')
    # Dimensions of the neural network
    input_output_dimension = 52
    encoding_dimension = 12
    intermediateLayer_dimension = 25

    dataSize = 10000  # The reduces size of the raw data

    # Import data
    rdata_read = pyreadr.read_r(
        "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
    all_df = rdata_read["faulty_training"]

    df = all_df.iloc[:dataSize, 3:]
    df = normalize_dataframe(df)

    # CREATE NEURAL NETWORK
    # Define input layer
    input_layer = keras.Input(shape=(input_output_dimension,))

    # Encoder
    encoder = layers.Dense(25, activation='relu')(input_layer)
    encoder = layers.Dense(encoding_dimension, activation='relu')(encoder)

    # Decoder
    decoder = layers.Dense(25, activation='relu')(encoder)
    decoder = layers.Dense(input_output_dimension,
                           activation='sigmoid')(decoder)

    # Create the autoencoder model
    autoencoder = keras.Model(inputs=input_layer, outputs=decoder)

    # Compile the model
    # autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    autoencoder.compile(optimizer='adam', loss='mse')

    #  prepare  input data.
    # first 2 thirds for training
    xTrain = df.iloc[:math.floor(dataSize*2/3), :]
    xTest = df.iloc[math.ceil(dataSize*2/3):, :]  # final third for testing

    print('xTest size', np.shape(xTest))

    # Train the data
    autoencoder.fit(xTrain, xTrain, epochs=50,
                    shuffle=True, validation_data=(xTest, xTest), )

    # Encode and decode some data
    predictedData = autoencoder.predict(xTest)

    # DISPLAY RESULTS

    # n = 5  # How many graphs to display
    # plt.figure(figsize=(20, 4))

    # for i in range(n):
    #     # Display original
    #     ax = plt.subplot(2, 5, i + 1)
    #     plt.plot(xTest.iloc[:, i])
    #     title = 'xmeas' + str(i)
    #     ax.set_title(title)

    #     # Display reconstruction
    #     ax = plt.subplot(2, 5, i + 1 + 5)
    #     plt.plot(predictedData[:, i])
    # plt.show()

    # TRANSFORM THE RAW DATA INTO LATENT SPACE
    encoded = keras.Model(inputs=autoencoder.input, outputs=encoder)

    # Get the latent space representation for the input data
    latent_space = encoded.predict(xTest)

    # Convert latent space to data frame
    latentSpaceDF = pd.DataFrame(latent_space)
    print('Latent space:', latentSpaceDF)

    fig = px.scatter_3d(latentSpaceDF, x=0, y=1, z=2)
    return fig


if __name__ == "__main__":
    app.run_server(debug=True)
