import pandas as pd
import pyreadr

rdata_read = pyreadr.read_r(
    'D:/T_Eastmen_Data/archive/TEP_Faulty_Training.RData')

df = rdata_read['faulty_training']
df = df.iloc[:300, :10]
df = pd.DataFrame(df)

# Specify the CSV file path where you want to save the DataFrame
csv_file = "TenesseeEastemen_FaultyTraining_Subsection.csv"

# Save the DataFrame to a CSV file
df.to_csv(csv_file, index=False)

print(f"DataFrame saved as {csv_file}")
