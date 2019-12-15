from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import numpy as np
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.model_selection import KFold


def random_forest_classifier_kfold_validation(df_gender):
    data = df_gender.to_numpy()
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    kf = KFold(n_splits=10)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf = RandomForestRegressor(n_estimators=10)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print("Error rate: " + str(mean_squared_error(y_test, y_pred)))
