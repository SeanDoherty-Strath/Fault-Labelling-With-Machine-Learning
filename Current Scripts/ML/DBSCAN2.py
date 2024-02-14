import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np
from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Load sample data (you can replace this with your own dataset)
# DATA
data = pd.read_csv("Data/UpdatedData.csv")
# data = data.drop(data.columns[[0, 1, 2, 3]], axis=1)  # Remove extra column
data = data.iloc[:, 4:]  # Remove extra column

scaler = StandardScaler()
scaledData = scaler.fit_transform(data)
X = pd.DataFrame(scaledData, columns=data.columns)


pca = PCA(n_components=10)
principal_components = pca.fit_transform(scaledData)

columns = []
for i in range(10):
    columns.append('PCA' + str(i))
principal_df = pd.DataFrame(data=principal_components, columns=columns)
print(principal_df)

# Initialize Dash app
app = dash.Dash(__name__)


def findBestParams():
    # Define a range of epsilon values and min_samples values to search
    eps_range = np.arange(1, 100, 10)
    min_samples_range = range(1, 11)

    best_score = -1
    best_eps = None
    best_min_samples = None

    # Perform grid search
    for eps_value in eps_range:
        for min_samples_value in min_samples_range:
            # Perform DBSCAN clustering
            dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
            cluster_labels = dbscan.fit_predict(X)

            if (len(list(set(cluster_labels))) != 1):

                # Compute silhouette score
                score = silhouette_score(X, cluster_labels)
                print(score)

            # Update best score and parameters if necessary
                if score > best_score:
                    best_score = score
                    best_eps = eps_value
                    best_min_samples = min_samples_value
    return best_eps, best_min_samples

    print("Best silhouette score:", best_score)
    print("Best eps:", best_eps)
    print("Best min_samples:", best_min_samples)


# Define layout of the app
app.layout = html.Div([
    html.H1("DBSCAN Clustering with Dash"),

    # Dropdown for selecting eps value
    html.Label("Select eps value for DBSCAN:"),
    dcc.Slider(
        id='eps-slider',
        min=0.1,
        max=10,
        step=0.5,
        value=0.1,
        # marks={i/10: str(i/10) for i in range(1, 21)},
    ),

    # Dropdown for selecting min_samples value
    html.Label("Select min_samples value for DBSCAN:"),
    dcc.Slider(
        id='min-samples-slider',
        min=1,
        max=10,
        step=1,
        value=3,

    ),

    # Scatter plot for displaying clustering result
    dcc.Graph(id='cluster-plot')
])

# Define callback for updating scatter plot based on slider values


@app.callback(
    Output('cluster-plot', 'figure'),
    [Input('eps-slider', 'value'),
     Input('min-samples-slider', 'value')]
)
def update_cluster_plot(eps_value, min_samples_value):
    # Perform DBSCAN clustering
    # best_eps, best_min_samples = findBestParams()

    # dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    dbscan.fit(principal_df)

    # Add cluster labels to the dataset
    cluster_labels = dbscan.labels_

    # Create a DataFrame with original data and cluster labels
    clustered_data = principal_df
    clustered_data['cluster'] = cluster_labels

    print(len(set(cluster_labels)))

    # Plot the clustered data
    fig = px.scatter_3d(clustered_data, x='PCA0', y='PCA1', z='PCA2',
                        color='cluster',
                        title=f'DBSCAN Clustering (eps={eps_value}, min_samples={min_samples_value})')

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
