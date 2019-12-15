import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score


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


def classifier_kfold_validation(df_gender, clf):
    """
    RandomForestClassifier
    Scores: [0.75892401 0.76454945 0.76577682 0.76639051 0.76237725]
    Accuracy: 0.76 (+/- 0.01)
    :param df_gender:
    :param clf:
    :return:
    """
    data = df_gender.to_numpy()
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    scores = cross_val_score(clf, X, y, cv=5)
    print("Scores: " + str(scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))


def classifier_kfoldn(df_gender):
    """
    RandomForestClassifier
    Scores: [0.75892401 0.76454945 0.76577682 0.76639051 0.76237725]
    Accuracy: 0.76 (+/- 0.01)
    :param df_gender:
    :param clf:
    :return:
    """
    data = df_gender.to_numpy()
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestClassifier(n_estimators=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Error rate: " + str(accuracy_score(y_test, y_pred)))
