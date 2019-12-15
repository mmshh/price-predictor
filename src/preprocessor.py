import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from src.classifiers import random_forest_classifier_kfold_validation
from src.utils import read_data_to_dataframe, apply_one_hot
import pandas as pd


def fill_missing_data(df):
    df.drop(["name", "host_name"], axis=1, inplace=True)
    df['last_review'] = df["last_review"].astype('datetime64[ns]')
    df['last_review'] = df['last_review'].fillna(pd.Timestamp("1900-1-1"))
    df = df.copy()
    # df['review_year'] = pd.DatetimeIndex(df['last_review']).year
    # df['review_month'] = pd.DatetimeIndex(df['last_review']).month
    # df['review_day'] = pd.DatetimeIndex(df['last_review']).day

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = apply_one_hot(df, "neighbourhood_group", "ng_")
    df = apply_one_hot(df, "room_type", "rt_")
    df = apply_one_hot(df, "neighbourhood", "nb_")
    df.drop(["neighbourhood_group", "room_type", "last_review", "neighbourhood"], axis=1, inplace=True)
    df.drop(["minimum_nights","calculated_host_listings_count","number_of_reviews" ,"reviews_per_month" ], axis=1, inplace=True)
    df = df[[c for c in df if c not in ['price']]
            + ['price']]
    return df


if __name__ == "__main__":
    df = read_data_to_dataframe("AB_NYC_2019.csv")
    df = fill_missing_data(df)
    # df["price"] = np.log(X + 1)
    # df["price"] = (X - X.min()) / (X.max() - X.min())
    df = df[df["price"] > 0]
    sns.distplot(df['price'])
    plt.show()
    random_forest_classifier_kfold_validation(df)
