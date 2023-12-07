import pyreadr
from pathlib import Path
import pandas as pd

rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
print(rdata_read)
df = rdata_read["faulty_training"]

print('DF: ', df)
print('DF Shape: ', df.shape)

filepath = Path('./FaultyTraining.csv')

df.to_csv(filepath)
