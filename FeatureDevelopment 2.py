import pandas as pd


def normalize_dataframe(df):
    """
    Normalize all values in a DataFrame between 0 and 1 using Min-Max normalization.

    Parameters:
    - df: pandas DataFrame

    Returns:
    - normalized_df: pandas DataFrame with normalized values
    """
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
