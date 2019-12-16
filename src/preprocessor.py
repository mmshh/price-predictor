import math

from sklearn.ensemble import RandomForestClassifier

from src.utils import read_data_to_dataframe, apply_one_hot, classifier_learn, classifier_kfold_validation
import pandas as pd


def fill_missing_data(df):
    df.drop(["name", "host_name"], axis=1, inplace=True)
    df['last_review'] = df["last_review"].astype('datetime64[ns]')
    df['last_review'] = df['last_review'].fillna(pd.Timestamp("1900-1-1"))
    df = df.copy()
    df['review_year'] = pd.DatetimeIndex(df['last_review']).year
    df['review_month'] = pd.DatetimeIndex(df['last_review']).month
    df['review_day'] = pd.DatetimeIndex(df['last_review']).day

    df["reviews_per_month"] = df["reviews_per_month"].fillna(0)
    df = apply_one_hot(df, "neighbourhood_group", "ng_")
    df = apply_one_hot(df, "room_type", "rt_")
    df = apply_one_hot(df, "neighbourhood", "nb_")
    df.drop(["neighbourhood_group", "room_type", "last_review", "neighbourhood"], axis=1, inplace=True)
    df.drop(["minimum_nights", "calculated_host_listings_count", "number_of_reviews", "reviews_per_month"], axis=1,
            inplace=True)
    df = df[[c for c in df if c not in ['price']]
            + ['price']]
    df = df[df["price"] > 0]
    df['price'] = pd.cut(df['price'], [0, 150, 300, 450, math.inf],
                         labels=["xx-150", "150-300", "300-450", "450-xx"],
                         right=False)
    return df


if __name__ == "__main__":
    df = read_data_to_dataframe("AB_NYC_2019.csv")
    df = fill_missing_data(df)
    clf = RandomForestClassifier()
    classifier_kfold_validation(df, clf)

    # run_exhaustive_search(clf, df, 1, parameter_space)
    # train_acc, test_acc =classifier_learn(df)
    #
    # data = [1, 5, 10, 25, 35, 50, 75, 100, 150, 175, 200]
    # from matplotlib import pyplot as plt
    # 
    # plt.plot(data, train_acc)
    # plt.plot(data, test_acc)
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('number of rounds')
    # plt.legend(['train', 'val'], loc='upper left')
    # plt.show()
