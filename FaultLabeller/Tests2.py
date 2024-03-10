from AutoLabellingFunctions import performKMeans, performPCA, performDBSCAN, findKneePoint, performAutoEncoding
import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output, State
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from dash.exceptions import PreventUpdate
import io
import base64  # Import base64 module
import time
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# DATA
data = pd.read_csv("FaultLabeller/Data/OperatingScenario1.csv")
data = data.drop(data.columns[[0]], axis=1)  # Remove extra columns
print(data)

zeros = [0] * 480
ones = [1] * 480
twos = [2] * 480
threes = [3] * 480

combinations = []
combinations.append(zeros + ones + zeros + twos + zeros + threes)
combinations.append(zeros + ones + zeros + threes + zeros + twos)
combinations.append(zeros + twos + zeros + ones + zeros + threes)
combinations.append(zeros + twos + zeros + threes + zeros + ones)
combinations.append(zeros + threes + zeros + ones + zeros + twos)
combinations.append(zeros + threes + zeros + twos + zeros + ones)

combinations.append(ones + threes + ones + twos + ones + zeros)
combinations.append(ones + threes + ones + zeros + ones + twos)
combinations.append(ones + twos + ones + zeros + ones + threes)
combinations.append(ones + twos + ones + threes + ones + zeros)
combinations.append(ones + zeros + ones + twos + ones + threes)
combinations.append(ones + zeros + ones + threes + ones + twos)

combinations.append(twos + zeros + twos + ones + twos + threes)
combinations.append(twos + zeros + twos + threes + twos + ones)
combinations.append(twos + ones + twos + zeros + twos + threes)
combinations.append(twos + ones + twos + threes + twos + zeros)
combinations.append(twos + threes + twos + zeros + twos + ones)
combinations.append(twos + threes + twos + ones + twos + zeros)

combinations.append(threes + zeros + threes + ones + threes + twos)
combinations.append(threes + zeros + threes + twos + threes + ones)
combinations.append(threes + ones + threes + zeros + threes + twos)
combinations.append(threes + ones + threes + twos + threes + ones)
combinations.append(threes + twos + threes + zeros + threes + ones)
combinations.append(threes + twos + threes + ones + threes + zeros)


bestAccuracy = 0
k = 13

data = performAutoEncoding(data)
# data = performPCA(data, 3)
eps = findKneePoint(data, k)
predictedLabels = performDBSCAN(data, eps, k)
print(eps)


for c in combinations:
    agreed_elements = 0
    for item1, item2 in zip(predictedLabels, c):
        if item1 == item2:
            agreed_elements += 1

    accuracy_percentage = (agreed_elements / len(c)) * 100

    if accuracy_percentage > bestAccuracy:
        bestAccuracy = accuracy_percentage


print('Accuracy: %:', bestAccuracy)
