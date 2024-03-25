from InternalLibraries.ML_Functions import performKMeans, performPCA, performDBSCAN, findKneePoint, createAutoencoder, trainNeuralNetwork, useNeuralNetwork
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
from sklearn.metrics import accuracy_score

# DATA

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


# # Scenario 2
# #   - Normal operation 100 samples
# #   - Fault 1 for 20 samples
# #   - Normal operation 100 samples
# #   - Fault 2 for 20 samples
# # - Normal operation 100 samples
# #   - Fault 3 for 20 samples
# #   - Repeated three times
# combinations = []

# zeros = [0] * 100
# ones = [1] * 20
# twos = [2] * 20
# threes = [3] * 20


# combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
#                     zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
# combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
#                     zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
# combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
#                     zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
# combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
#                     zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
# combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
#                     zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
# combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
#                     zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

# zeros = [1] * 100
# ones = [0] * 20
# twos = [2] * 20
# threes = [3] * 20


# combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
#                     zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
# combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
#                     zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
# combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
#                     zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
# combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
#                     zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
# combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
#                     zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
# combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
#                     zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

# zeros = [2] * 100
# ones = [1] * 20
# twos = [0] * 20
# threes = [3] * 20


# combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
#                     zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
# combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
#                     zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
# combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
#                     zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
# combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
#                     zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
# combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
#                     zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
# combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
#                     zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)

# zeros = [3] * 100
# ones = [1] * 20
# twos = [2] * 20
# threes = [0] * 20


# combinations.append(zeros + ones + zeros + twos + zeros + threes + zeros + ones +
#                     zeros + twos + zeros + threes + zeros + ones + zeros + twos + zeros + threes)
# combinations.append(zeros + ones + zeros + threes + zeros + twos + zeros + ones +
#                     zeros + threes + zeros + twos + zeros + ones + zeros + threes + zeros + twos)
# combinations.append(zeros + twos + zeros + ones + zeros + threes + zeros + twos +
#                     zeros + ones + zeros + threes + zeros + twos + zeros + ones + zeros + threes)
# combinations.append(zeros + twos + zeros + threes + zeros + ones + zeros + twos +
#                     zeros + threes + zeros + ones + zeros + twos + zeros + threes + zeros + ones)
# combinations.append(zeros + threes + zeros + ones + zeros + twos + zeros + threes +
#                     zeros + ones + zeros + twos + zeros + threes + zeros + ones + zeros + twos)
# combinations.append(zeros + threes + zeros + twos + zeros + ones + zeros + threes +
#                     zeros + twos + zeros + ones + zeros + threes + zeros + twos + zeros + ones)


# # Scenario 3
# #   - Normal operation for 480 samples
# #   - Fault 3 for 300 samples
# #   - Fault 1 for 300 samplels
# #   - Fault 2 for 300 samples
# # Repeated twice times
# combinations = []
# zeros = [0] * 480
# ones = [1] * 300
# twos = [2] * 300
# threes = [3] * 300
# combinations.append(zeros + ones + twos + threes)
# combinations.append(zeros + ones + threes + twos)
# combinations.append(zeros + twos + ones + threes)
# combinations.append(zeros + twos + threes + ones)
# combinations.append(zeros + threes + ones + twos)
# combinations.append(zeros + threes + twos + ones)


# zeros = [1] * 480
# ones = [0] * 300
# twos = [2] * 300
# threes = [3] * 300
# combinations.append(zeros + ones + twos + threes)
# combinations.append(zeros + ones + threes + twos)
# combinations.append(zeros + twos + ones + threes)
# combinations.append(zeros + twos + threes + ones)
# combinations.append(zeros + threes + ones + twos)
# combinations.append(zeros + threes + twos + ones)


# zeros = [2] * 480
# ones = [1] * 300
# twos = [0] * 300
# threes = [3] * 300
# combinations.append(zeros + ones + twos + threes)
# combinations.append(zeros + ones + threes + twos)
# combinations.append(zeros + twos + ones + threes)
# combinations.append(zeros + twos + threes + ones)
# combinations.append(zeros + threes + ones + twos)
# combinations.append(zeros + threes + twos + ones)

# zeros = [3] * 480
# ones = [1] * 300
# twos = [2] * 300
# threes = [0] * 300
# combinations.append(zeros + ones + twos + threes)
# combinations.append(zeros + ones + threes + twos)
# combinations.append(zeros + twos + ones + threes)
# combinations.append(zeros + twos + threes + ones)
# combinations.append(zeros + threes + ones + twos)
# combinations.append(zeros + threes + twos + ones)


bestAccuracy = 0
# k = 8

trainingData = pd.read_csv("FaultLabeller/Data/Scenario5withLabels.csv")
trainingData = trainingData.drop(
    trainingData.columns[[0]], axis=1)  # Remove extra columns

testData = pd.read_csv("FaultLabeller/Data/Scenario3withLabels.csv")
testData = testData.drop(testData.columns[[0]], axis=1)  # Remove extra columns
correctLabels = testData.iloc[:, -1]
# print(correctLabels)
# Remove extra columns
testData = testData.drop(testData.columns[[-1]], axis=1)

# data = performPCA(data, 15)

startTime = time.time()
# autoencoder = createAutoencoder(data)
NN = trainNeuralNetwork(trainingData)
predictLabels = useNeuralNetwork(testData, NN)

endTime = time.time() - startTime
# # data = performPCA(data, 3)
# eps = findKneePoint(data, k)
# predictedLabels = performDBSCAN(data, eps+0.1, k)
# predictedLabels = performKMeans(data, 4)
# print(eps)


# for c in combinations:
#     agreed_elements = 0
#     for item1, item2 in zip(predictedLabels, c):
#         if item1 == item2:
#             agreed_elements += 1

#     accuracy_percentage = (agreed_elements / len(c)) * 100

#     if accuracy_percentage > bestAccuracy:
#         bestAccuracy = accuracy_percentage


score = 0


for i in range(len(predictLabels)):
    if predictLabels[i] == correctLabels[i]:
        score += 1

accuracy = score / len(predictLabels) * 100
print('Accuracy = ', accuracy)

print('Time:', endTime)
#
