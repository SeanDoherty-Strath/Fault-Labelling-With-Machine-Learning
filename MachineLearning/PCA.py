import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

df = pd.read_csv("TenesseeEastemen_FaultyTraining_Subsection.csv")
df = df.iloc[:, 3:8]

# To INFER the number of important components:
# pca = PCA(n_components="mle", svd_solver="full")

# To SET the number of components:
pca = PCA(n_components=3)

pca.fit(df)

print("N featuers before: ", pca.n_features_in_)
print("N components after: ", pca.n_components_)

print("Explained variance ratio", pca.explained_variance_ratio_)
print(pca.singular_values_)
