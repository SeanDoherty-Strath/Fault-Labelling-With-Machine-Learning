import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn import datasets
import numpy as np
from sklearn.metrics import silhouette_score

# Load sample data (you can replace this with your own dataset)
iris = datasets.load_iris()
X = iris.data

# Initialize Dash app
app = dash.Dash(__name__)


def findBestParams():
    # Define a range of epsilon values and min_samples values to search
    eps_range = np.arange(0.1, 2.1, 0.1)
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

            print(cluster_labels)
            print(len(list(set(cluster_labels))))
            if (len(list(set(cluster_labels))) != 1):

                # Compute silhouette score
                score = silhouette_score(X, cluster_labels)

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
        max=2.0,
        step=0.1,
        value=0.5,
        marks={i/10: str(i/10) for i in range(1, 21)},
    ),

    # Dropdown for selecting min_samples value
    html.Label("Select min_samples value for DBSCAN:"),
    dcc.Slider(
        id='min-samples-slider',
        min=1,
        max=10,
        step=1,
        value=3,
        marks={i: str(i) for i in range(1, 11)},
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
    best_eps, best_min_samples = findBestParams()

    dbscan = DBSCAN(eps=best_eps, min_samples=best_min_samples)
    dbscan.fit(X)

    # Add cluster labels to the dataset
    cluster_labels = dbscan.labels_

    # Create a DataFrame with original data and cluster labels
    clustered_data = px.data.iris()
    clustered_data['cluster'] = cluster_labels

    # Plot the clustered data
    fig = px.scatter_3d(clustered_data, x='sepal_length', y='sepal_width', z='petal_width',
                        color='cluster', symbol='species',
                        title=f'DBSCAN Clustering (eps={eps_value}, min_samples={min_samples_value})')

    return fig


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
