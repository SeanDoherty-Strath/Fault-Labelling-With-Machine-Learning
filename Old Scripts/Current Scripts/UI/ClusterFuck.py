import random
import plotly.express as px
import pandas as pd
from dash import dcc
import math
import statistics
import numpy as np


array = []


#  This is one cluster
for i in range(1000):
    x = random.randint(50, 100)
    y = random.randint(50,100)
    z = random.randint(0, 50)
    w = random.randint(0, 50)
    v = random.randint(0, 20)
    array.append([x, y, z, w, v])

df = pd.DataFrame(array)


# Calculate centroid
centroid = df.mean()
print(centroid)

# Calculate Euclidean distance for each point in the cluster
df['Euclidean 1'] = np.sqrt(np.sum((df[[0]] - centroid)**2, axis=1))
df['Euclidean 2'] = np.sqrt(np.sum((df[[1]] - centroid)**2, axis=1))
df['Euclidean 3'] = np.sqrt(np.sum((df[[2]] - centroid)**2, axis=1))


avg1 = df['Euclidean 1'].mean()
avg2 = df['Euclidean 2'].mean()
avg3 = df['Euclidean 3'].mean()

print(avg1, avg2, avg3)