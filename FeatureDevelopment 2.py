import pandas as pd


def normalize_dataframe(df):

    normalized_df = (df - df.min()) / (df.max() - df.min())
    return normalized_df


# Example usage:
# Assuming df is your DataFrame
data = {
    'Feature1': [10, 20, 15, 8, 25],
    'Feature2': [5, 15, 8, 12, 18]
}

df = pd.DataFrame(data)

normalized_df = normalize_dataframe(df)
print(normalized_df)
