from src.utils import read_data_to_dataframe
import pandas as pd


def fill_missing_data(df):
    df.drop(["name", "host_name"], axis=1, inplace=True)
    df['last_review'] = df["last_review"].astype('datetime64[ns]')
    df['last_review'] = df['last_review'].fillna(pd.Timestamp("1900-1-1"))
    df["reviews_per_month"] =df["reviews_per_month"].fillna(0)
    return df


if __name__ == "__main__":
    df =read_data_to_dataframe("AB_NYC_2019.csv")
    fill_missing_data(df)
