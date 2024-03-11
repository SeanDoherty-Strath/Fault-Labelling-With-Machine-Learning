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


operatingScenario = 7

if operatingScenario == 1:

    # No fault, 24 hours
    newDF = FaultFreeData.iloc[:480, :]
    # correctLabels = pd.DataFrame([0]*480)
    newDF = newDF._append(FaultyData.iloc[20:500], ignore_index=True)

    newDF = newDF._append(FaultFreeData.iloc[500:980], ignore_index=True)
    # correctLabels = correctLabels._append([0]*480)
    newDF = newDF._append(FaultyData.iloc[520:1000], ignore_index=True)
    # correctLabels = correctLabels._append([2]*480)
    newDF = newDF._append(FaultFreeData.iloc[1000:1480], ignore_index=True)
    # correctLabels = correctLabels._append(10*480)
    newDF = newDF._append(
        FaultyData.iloc[5*480+20:5*480+500], ignore_index=True)

    # correctLabels = correctLabels._append([3]*480)

    # correctLabels = [[0]*480, [1]*480, [0]*480, [2]*480, [0]*480, [3]*480]
    # correct_labels = pd.DataFrame(correctLabels)
    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
    # newDF = newDF.rename(columns={'Unnamed: 0': 'Time'})  # Rename First Column

    print(newDF.shape)
    # print(correctLabels.shape)

    filepath = Path('FaultLabeller/Data/OperatingScenario1.csv')
    newDF.to_csv(filepath)
    # filepath = Path('FaultLabeller/Data/CorrectLabelsScenario1.csv')
    # correctLabels.to_csv(filepath)

if operatingScenario == 2:

    # No fault, 24 hours
    newDF = FaultFreeData.iloc[1500:2220, :]
    newDF = newDF._append(FaultyData.iloc[2020:2260], ignore_index=True)
    newDF = newDF._append(FaultFreeData.iloc[2220:2700], ignore_index=True)

    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
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

if operatingScenario == 4:
    newDF = FaultFreeData.iloc[:480, :]
    for i in range(0, 20):
        newDF = newDF._append(
            FaultyData.iloc[i*480+20:i*480+500])
    print(newDF.shape)
    # expecting 10080
    filepath = Path('FaultLabeller/Data/AllFaults.csv')
    newDF.to_csv(filepath)
if operatingScenario == 5:
    # newDF = [FaultFreeData.iloc[:480, :]]
    newDF = pd.DataFrame()
    for i in range(0, 10):
        # data set repeats every 500
        newDF = newDF._append(FaultFreeData.iloc[500*i:480+500*i, :])
        # data set repeats every
        newDF = newDF._append(FaultyData.iloc[i*10000+20:i*10000+500])

    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
    print(newDF.shape)
    print(newDF)
    # expecting 9600
    filepath = Path('FaultLabeller/Data/OperatingScenario5.csv')
    newDF.to_csv(filepath)

if operatingScenario == 6:
    # newDF = [FaultFreeData.iloc[:480, :]]
    newDF = pd.DataFrame()
    for i in range(0, 10):
        # data set repeats every 500
        newDF = newDF._append(FaultFreeData.iloc[500*i:480+500*i, :])
        # data set repeats every 10000
        newDF = newDF._append(
            FaultyData.iloc[i*10000+20:i*10000+500])  # Fault 2
        newDF = newDF._append(
            FaultyData.iloc[i*10000+20+500:i*10000+500+500])  # Fault 2
        newDF = newDF._append(
            FaultyData.iloc[i*10000+20+5*500:i*10000+500+500*5])  # Fault 3
    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
    y = [0]*480 + [1]*480 + [2]*480 + [3]*480
    newDF['labels'] = y*10
    print(newDF.shape)
    print(newDF)

    # expecting 19200
    filepath = Path('FaultLabeller/Data/OperatingScenario6WithLabels.csv')
    newDF.to_csv(filepath)

if operatingScenario == 7:
    # newDF = [FaultFreeData.iloc[:480, :]]
    newDF = pd.DataFrame()
    for i in range(0, 5):
        # data set repeats every 500
        newDF = newDF._append(FaultFreeData.iloc[500*i:480+500*i, :])
        # data set repeats every 10000
        newDF = newDF._append(
            FaultyData.iloc[i*10000+20:i*10000+200])  # Fault 1
        newDF = newDF._append(
            FaultyData.iloc[i*10000+20+5*500:i*10000+500+500*5])  # Fault 3
        newDF = newDF._append(
            FaultyData.iloc[i*10000+20+500:i*10000+200+500])  # Fault 2
    newDF = newDF.drop(data.columns[[0, 1, 2]], axis=1)  # Remove extra columns
    y = [0]*480 + [1]*180 + [3]*480 + [2]*180
    newDF['labels'] = y*5
    print(newDF.shape)
    print(newDF)

    filepath = Path('FaultLabeller/Data/OperatingScenario7WithLabels.csv')
    newDF.to_csv(filepath)
