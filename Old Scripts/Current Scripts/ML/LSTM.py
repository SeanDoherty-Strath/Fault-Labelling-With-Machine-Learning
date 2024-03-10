import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from keras.models import Sequential
from keras.layers import LSTM, Dense

# Generate sample data (replace with your data)
np.random.seed(0)
data = np.random.rand(2000, 52)
df = pd.DataFrame(data, columns=[f'Feature_{i}' for i in range(52)])

# Preprocessing
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# LSTM parameters
n_steps = 4  # Number of time steps
n_features = 52  # Number of features

# Reshape data for LSTM
X = scaled_data.reshape(-1, n_steps, n_features)
print('X SHAPE: ', X.shape)

# LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(n_features))
model.compile(optimizer='adam', loss='mse')
model.summary()
# model.fit(X, X, epochs=200, verbose=0)

# K-Means clustering
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(model.predict(X))

# Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Graph(id='lstm-plot'),
    dcc.Graph(id='kmeans-plot')
])


@app.callback(
    [Output('lstm-plot', 'figure'),
     Output('kmeans-plot', 'figure')],
    [Input('lstm-plot', 'clickData')]
)
def update_plots(clickData):
    # LSTM compressed data
    lstm_compressed = model.predict(X)

    # Plot original vs compressed data
    original_vs_compressed = pd.DataFrame({
        'Time': range(len(df)),
        'Original': scaled_data.flatten(),
        'Compressed': lstm_compressed.flatten()
    })

    lstm_fig = px.line(original_vs_compressed, x='Time', y=['Original', 'Compressed'],
                       title='Original vs Compressed Data', labels={'value': 'Value'})

    # Plot K-Means clusters
    kmeans_fig = px.scatter(x=range(len(clusters)), y=lstm_compressed.flatten(), color=clusters,
                            title='K-Means Clusters', labels={'x': 'Time', 'y': 'Compressed Value'})

    return lstm_fig, kmeans_fig


if __name__ == '__main__':
    app.run_server(debug=True)
