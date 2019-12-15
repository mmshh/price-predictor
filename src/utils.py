import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_data_to_dataframe(path):
    print(os.path.abspath(__file__))
    return pd.read_csv(path)


@staticmethod
def split_data(df):
    data = df.to_numpy()
    X = data[:, :-1]
    y = data[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return X_train, X_test, y_train, y_test


def apply_one_hot(df, column_name, prefix):
    return pd.concat([df, pd.get_dummies(df[column_name], prefix=prefix)], axis=1)


def evaluate(model, test_features, test_labels):
    predictions = model.predict(test_features)
    errors = abs(predictions - test_labels)
    mape = 100 * np.mean(errors / test_labels)
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))

    return accuracy