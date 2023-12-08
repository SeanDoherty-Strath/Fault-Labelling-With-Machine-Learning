import pyreadr
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

x = 'test'

if x == 'test':
    rdata_read = pyreadr.read_r(
        "D:/T_Eastmen_Data/archive/TEP_Faulty_Testing.RData")
    print(rdata_read)
    data = rdata_read["faulty_testing"]
    df = pd.DataFrame(data)

    rdata_read = pyreadr.read_r(
        "D:/T_Eastmen_Data/archive/TEP_FaultFree_Testing.RData")
    data = rdata_read["fault_free_testing"]
    df2 = pd.DataFrame(data)

    print(df)
    print(df2)
    print(df.shape)
    print(df2.shape)

    # No fault
    newDF = df2.iloc[:960*10, :]

    for i in range(10):

        # Fault 1
        newDF = newDF._append(df.iloc[19200*i:960+19200*i], ignore_index=True)
        # Fault 2
        newDF = newDF._append(
            df.iloc[19200*i+960:960*2+19200*i], ignore_index=True)
        # Fault 3
        newDF = newDF._append(
            df.iloc[19200*i+960*2:960*3+19200*i], ignore_index=True)

    print(newDF)
    print(newDF.shape)
    # newDF = newDF.append(df.iloc[], ignore_index=True)

    # PLOT SOME OF THE DATA
    # Plotting first five faults
    # plt.figure(figsize=(20, 16))

    # for sensor in range(5):  # sensors 1- 5
    #     for fault in range(4):  # faults 0, 1, 2 and 3
    #         # four Faults along the top, five sensors down the side
    #         ax = plt.subplot(5, 4, fault+1+sensor*4)
    #         data = newDF.iloc[fault * 500:fault*500+500, sensor+3]
    #         print(data)
    #         plt.plot(data)
    #         title = 'Fault ' + str(fault), ', Sensor ' + str(sensor)

    #         ax.set_title(title)

    # plt.show()

    filepath = Path('./UpdatedTestData.csv')
    newDF.to_csv(filepath)

elif (x == 'train'):
    rdata_read = pyreadr.read_r(
        "D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData")
    data = rdata_read["faulty_training"]
    df = pd.DataFrame(data)

    rdata_read = pyreadr.read_r(
        "D:/T_Eastmen_Data/archive/TEP_FaultFree_Training.RData")
    data = rdata_read["fault_free_training"]
    df2 = pd.DataFrame(data)

    print(df)
    print(df2)
    print(df.shape)
    print(df2.shape)

    # No fault
    newDF = df2.iloc[:500*10, :]

    for i in range(10):

        # Fault 1
        newDF = newDF._append(df.iloc[10000*i:500+10000*i], ignore_index=True)
        # Fault 2
        newDF = newDF._append(
            df.iloc[10000*i+500:1000+10000*i], ignore_index=True)
        # Fault 3
        newDF = newDF._append(
            df.iloc[10000*i+1000:1500+10000*i], ignore_index=True)

    print(newDF)
    print(newDF.shape)
    # newDF = newDF.append(df.iloc[], ignore_index=True)

    # PLOT SOME OF THE DATA
    # Plotting first five faults
    plt.figure(figsize=(20, 16))

    for sensor in range(5):  # sensors 1- 5
        for fault in range(4):  # faults 0, 1, 2 and 3
            # four Faults along the top, five sensors down the side
            ax = plt.subplot(5, 4, fault+1+sensor*4)
            data = newDF.iloc[fault * 500:fault*500+500, sensor+3]
            print(data)
            plt.plot(data)
            title = 'Fault ' + str(fault), ', Sensor ' + str(sensor)

            ax.set_title(title)

    plt.show()

    filepath = Path('./UpdatedData.csv')
    newDF.to_csv(filepath)
