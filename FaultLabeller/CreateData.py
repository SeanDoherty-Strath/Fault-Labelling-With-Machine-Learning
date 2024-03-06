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


# OPERATING SCENARRIOS
# Scenario 1
#   - Normal operation for 480 samples
#   - Fault 1 for 480 samples
#   - Normal operation 480 samples
#   - Fault 2 for 480 samplels
#   - Normal operation for 480 samples
#   - Fault 4 for 480 samples


# Scenario 2
#   - Normal operation 720 samples
#   - Fault 5 for 240 samples
#   - Normal operation for 480 samples

# Scenario 3
#   - Normal operation for 480 samples
#   - Fault 1 for 480 samples
#   - Normal operation 480 samples
#   - Fault 1 for 480 samplels
#   - Normal operation for 480 samples
#   - Fault 1 for 480 samples


operatingScenario = 3

if operatingScenario == 1:

    # No fault, 24 hours
    newDF = FaultFreeData.iloc[:480, :]
    newDF = newDF._append(FaultyData.iloc[20:500], ignore_index=True)
    newDF = newDF._append(FaultFreeData.iloc[500:980], ignore_index=True)
    newDF = newDF._append(FaultyData.iloc[520:1000], ignore_index=True)
    newDF = newDF._append(FaultFreeData.iloc[1000:1480], ignore_index=True)
    newDF = newDF._append(FaultyData.iloc[1520:2000], ignore_index=True)

    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
    newDF = newDF.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column

    print(newDF.shape)

    filepath = Path('FaultLabeller/Data/OperatingScenario1.csv')
    newDF.to_csv(filepath)

if operatingScenario == 2:

    # No fault, 24 hours
    newDF = FaultFreeData.iloc[1500:2220, :]
    newDF = newDF._append(FaultyData.iloc[2020:2260], ignore_index=True)
    newDF = newDF._append(FaultFreeData.iloc[2220:2700], ignore_index=True)

    newDF = newDF.drop(data.columns[[1, 2, 3]], axis=1)  # Remove extra columns
    newDF = newDF.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column

    print(newDF.shape)
    # Expecting 1440 by 52

    filepath = Path('FaultLabeller/Data/OperatingScenario2.csv')
    newDF.to_csv(filepath)

if operatingScenario == 3:

    # No fault, 24 hours
    newDF = FaultFreeData.iloc[:480, :]
    newDF = newDF._append(FaultyData.iloc[20:500], ignore_index=True)
    newDF = newDF._append(FaultFreeData.iloc[500:980], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[10000+20:10000+20+480], ignore_index=True)
    newDF = newDF._append(FaultFreeData.iloc[1000:1480], ignore_index=True)
    newDF = newDF._append(
        FaultyData.iloc[20000+20:20000+20+480], ignore_index=True)

    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
    newDF.columns.values[0] = 'Time'

    # newDF = newDF.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column

    print(newDF.shape)

    filepath = Path('FaultLabeller/Data/OperatingScenario3.csv')
    newDF.to_csv(filepath)
