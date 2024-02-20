import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import tensorflow as tf

app = dash.Dash(__name__)

# Define GRU autoencoder model


class GRUAutoencoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(GRUAutoencoder, self).__init__()
        self.encoder = tf.keras.layers.GRU(latent_dim, return_sequences=True)
        self.decoder = tf.keras.layers.GRU(latent_dim, return_sequences=True)
        self.latent_dim = latent_dim

    def call(self, inputs):
        encoded = self.encoder(inputs)
        decoded = self.decoder(encoded)
        return decoded


# Generate dummy data
latent_dim = 32
input_shape = (10, 32)
data = np.random.rand(1000, *input_shape).astype(np.float32)

# Create GRU autoencoder model
autoencoder = GRUAutoencoder(latent_dim)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(data, data, epochs=10, batch_size=32)

# Dash layout
app.layout = html.Div([
    html.H1("GRU Autoencoder Demo"),
    dcc.Graph(id='original-data-graph'),
    dcc.Graph(id='reconstructed-data-graph')
])

# Dash callbacks


@app.callback(
    [Output('original-data-graph', 'figure'),
     Output('reconstructed-data-graph', 'figure')],
    [Input('interval-component', 'n_intervals')]
)
def update_graph(n):
    # Generate new data for demonstration
    test_data = np.random.rand(1, *input_shape).astype(np.float32)

    # Encode and decode the test data
    encoded_data = autoencoder.encoder(test_data)
    reconstructed_data = autoencoder.decoder(encoded_data)

    # Original data figure
    original_fig = {
        'data': [{'x': np.arange(input_shape[0]), 'y': test_data[0, :, i], 'type': 'scatter', 'name': f'Feature {i+1}'}
                 for i in range(input_shape[1])],
        'layout': {'title': 'Original Data'}
    }

    # Reconstructed data figure
    reconstructed_fig = {
        'data': [{'x': np.arange(input_shape[0]), 'y': reconstructed_data[0, :, i], 'type': 'scatter', 'name': f'Feature {i+1}'}
                 for i in range(input_shape[1])],
        'layout': {'title': 'Reconstructed Data'}
    }

    return original_fig, reconstructed_fig


if __name__ == '__main__':
    app.run_server(debug=True)
