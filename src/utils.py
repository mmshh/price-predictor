import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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
    with test: 0.7582080392758515
    Accuracy: 0.76 (+/- 0.01)

    GradientBoostingClassifier
    Scores: [0.74823602 0.74514216 0.75306874 0.75153437 0.74915601]
    with test: 25 rounds: 0.7592308479083564, 50 rounds: 0.758719443592104
    Accuracy: 0.75 (+/- 0.01)

    SVC
    Scores: [0.44994376 0.35048067 0.14944763 0.64065057 0.5483376 ]
    Accuracy: 0.43 (+/- 0.34)
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


def classifier_learn(df):
    df = df.to_numpy()
    X = df[:, :-1]
    y = df[:, -1]
    x, x_test, y, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)
    x_train, x_cv, y_train, y_cv = train_test_split(x, y, test_size=0.25, train_size=0.75)
    clf = GradientBoostingClassifier(learning_rate=0.15, n_estimators=50, verbose=True)
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(accuracy_score(y_test, y_pred))

    test_acc = []
    train_acc = []
    # for e in [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 1]:
    #     clf = GradientBoostingClassifier(learning_rate=e, verbose=True)
    #     clf.fit(x_train, y_train)
    #     y_pred = clf.predict(x_cv)
    #     train_pred = clf.predict(x_train)
    #     test_acc.append(accuracy_score(y_cv, y_pred))
    #     train_acc.append(accuracy_score(y_train, train_pred))
    print("Test Acc rate: " + str(test_acc))
    print("Train Acc rate: " + str(train_acc))
    return train_acc, test_acc, x_test, y_test
