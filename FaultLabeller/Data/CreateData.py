import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
data = rdata_read["faulty_training"]
FaultyData = pd.DataFrame(data)

rdata_read = pyreadr.read_r(
    "D:/T_Eastmen_Data/archive/TEP_FaultFree_Training.RData")
data = rdata_read["fault_free_training"]
FaultFreeData = pd.DataFrame(data)


# SCENARIO 1
#   - Normal operation for 480 samples
#   - Fault 1 for 480 samples
#   - Normal operation 480 samples
#   - Fault 2 for 480 samplels
#   - Normal operation for 480 samples
#   - Fault 3 for 480 samples


newDF = FaultFreeData.iloc[:480, :]
newDF = newDF._append(FaultyData.iloc[20:500], ignore_index=True)
newDF = newDF._append(FaultFreeData.iloc[500:980], ignore_index=True)
newDF = newDF._append(FaultyData.iloc[520:1000], ignore_index=True)
newDF = newDF._append(FaultFreeData.iloc[1000:1480], ignore_index=True)
newDF = newDF._append(FaultyData.iloc[5*480+20:5*480+500], ignore_index=True)

newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
print(newDF.shape)
# expected 2880
filepath = Path('FaultLabeller/Data/Scenario1.csv')
newDF.to_csv(filepath)

labels = [1]*480 + [2]*480 + [1]*480 + [3]*480 + [1]*480 + [4]*480
print(labels)
newDF['correctLabels'] = labels
filepath = Path('FaultLabeller/Data/Scenario1withLabels.csv')
newDF.to_csv(filepath)


# Scenario 2
#   - Normal operation 100 samples
#   - Fault 1 for 20 samples
#   - Normal operation 100 samples
#   - Fault 2 for 20 samples
# - Normal operation 100 samples
#   - Fault 3 for 20 samples
#   - Repeated three times

newDF = pd.DataFrame()

for i in range(0, 3):
    # TE dataset repeats every 10 000
    faultFreeOffset = i*500
    faultyOffset = i*10000

    newDF = newDF._append(
        FaultFreeData.iloc[faultFreeOffset:faultFreeOffset+100, :])
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20:40+faultyOffset], ignore_index=True)
    newDF = newDF._append(
        FaultFreeData.iloc[faultFreeOffset+100:faultFreeOffset+200, :])
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20+500:40+faultyOffset+500], ignore_index=True)
    newDF = newDF._append(
        FaultFreeData.iloc[faultFreeOffset+200:faultFreeOffset+300, :])
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20+500*5:40+faultyOffset+500*5], ignore_index=True)


newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
print(newDF.shape)
# expected 1080
filepath = Path('FaultLabeller/Data/Scenario2.csv')
newDF.to_csv(filepath)

labels = [1]*100 + [2]*20 + [1]*100 + [3]*20 + [1]*100 + [4]*20
newDF['correctLabels'] = labels*3
filepath = Path('FaultLabeller/Data/Scenario2withLabels.csv')
newDF.to_csv(filepath)


# Scenario 3
#   - Normal operation for 480 samples
#   - Fault 3 for 300 samples
#   - Fault 1 for 300 samplels
#   - Fault 2 for 300 samples
# Repeated twice times

newDF = pd.DataFrame()

for i in range(0, 2):

    faultFreeOffset = i*500
    faultyOffset = i*10000

    newDF = newDF._append(
        FaultFreeData.iloc[faultFreeOffset:faultFreeOffset+480, :])
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20+500*5:320+faultyOffset+500*5], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20:320+faultyOffset], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20+500:320+faultyOffset+500], ignore_index=True)


newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
print(newDF.shape)
# expected 2760
filepath = Path('FaultLabeller/Data/Scenario3.csv')
newDF.to_csv(filepath)

labels = [1]*480 + [4]*300 + [2]*300 + [3]*300
newDF['correctLabels'] = labels*2
filepath = Path('FaultLabeller/Data/Scenario3withLabels.csv')
newDF.to_csv(filepath)


# Scenario 4
#   - Normal operation for 1440 samples
#   - Fault 2 for 256 samples
#   - Normal operation for 2160 samplels
#   - Fault 1  for 256 samples
# - Fault 3 for 256 samples


newDF = pd.DataFrame()

faultyOffset = 30000


newDF = newDF._append(FaultFreeData.iloc[faultyOffset:faultyOffset+1440, :])
newDF = newDF._append(
    FaultyData.iloc[faultyOffset+20+500:faultyOffset+20+500+256], ignore_index=True)
newDF = newDF._append(
    FaultFreeData.iloc[faultyOffset+1440:faultyOffset+1440+2160, :])
newDF = newDF._append(
    FaultyData.iloc[faultyOffset+20:faultyOffset+20+256], ignore_index=True)
newDF = newDF._append(
    FaultyData.iloc[faultyOffset+20+500*5:faultyOffset+20+256+500*5], ignore_index=True)


newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
print(newDF.shape)
# expected 4368
filepath = Path('FaultLabeller/Data/Scenario4.csv')
newDF.to_csv(filepath)

newDF['correctLabels'] = [1]*1440 + [3]*256 + [1]*2160 + [2]*256 + [3]*256
filepath = Path('FaultLabeller/Data/Scenario4withLabels.csv')
newDF.to_csv(filepath)


# Scenario 5
#   - Normal operation for 480 samples
#   - Fault 1 for 480 samples
#   - Fault 2  for 480 samples
#   - Fault 3 for 480 samples
#  Repeated 3 steps

newDF = pd.DataFrame()

for i in range(0, 3):
    print(i)
    faultFreeOffset = i*500
    faultyOffset = i*10000

    newDF = newDF._append(
        FaultFreeData.iloc[faultFreeOffset:faultFreeOffset+480], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20:faultyOffset+500], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20+500:faultyOffset+1000], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[faultyOffset+20+500*5:faultyOffset+500*6], ignore_index=True)


newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
print(newDF.shape)
# expected 25760
filepath = Path('FaultLabeller/Data/Scenario5.csv')
newDF.to_csv(filepath)

labels = [1]*480 + [2]*480 + [3]*480 + [4]*480
newDF['correctLabels'] = labels*3
filepath = Path('FaultLabeller/Data/Scenario5withLabels.csv')
newDF.to_csv(filepath)
