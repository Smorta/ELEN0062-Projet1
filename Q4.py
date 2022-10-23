import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from data import make_dataset2, make_dataset1
from sklearn.metrics import accuracy_score


def run_cross_validation(data, clf_type, hp_list, cv=5):
    """Return the best cross-validation score and the associated hyper parameter
     of the selected model on the dataset.
    Parameters
    ----------
    data : array-like, shape = [n_samples, n_features + 1]
            The dataset
    clf_type : str, "dt" or "knn"
            The type of model use
    Returns
    -------
    test_score : float
            The mean cross-validation score of the model on the dataset
    opt_hp : int
            The optimal value of the hyper parameter for the model on the dataset
    """
    if clf_type != "dt" and clf_type != "knn":
        raise Exception("the clf must be a 'dt' or 'knn'")
    if type(clf_type) is not str:
        raise Exception("the clf must be a string")

    sizeLS = 1200
    clf = None
    X, y = data
    # splitting the data set into learning set and testing set
    XLS = X[:sizeLS]
    yLS = y[:sizeLS]
    XTS = X[sizeLS:]
    yTS = y[sizeLS:]

    # first approximation of the optimal parameter
    mean_cv = np.zeros(len(hp_list))
    std_cv = np.zeros(len(hp_list))
    for i, hp in enumerate(hp_list):
        if clf_type == "dt":
            clf = DecisionTreeClassifier(max_depth=hp)
        else:
            clf = KNeighborsClassifier(n_neighbors=hp)

        score = cross_val_score(clf, XLS, yLS, cv=cv)
        mean_cv[i] = score.mean()
        std_cv[i] = score.std()

    opt_param_index = np.argmax(mean_cv)
    opt_hp = hp_list[opt_param_index]

    # finding the optimal parameter more precisely,
    # focussing on the area of hp's near current maximum
    if hp_list[-1] - hp_list[0] > len(hp_list):
        low_hp = 0
        high_hp = 10000
        if opt_param_index > 0:
            low_hp = hp_list[opt_param_index - 1]
        else:
            low_hp = hp_list[opt_param_index]
        if opt_param_index < len(hp_list) - 1:
            high_hp = hp_list[opt_param_index + 1]
        else:
            high_hp = hp_list[opt_param_index]
        new_hp = np.linspace(low_hp, high_hp, 8).astype(int)
        new_test_score, newopt_hp = run_cross_validation(data, clf_type, new_hp)
        opt_hp = newopt_hp

    if clf_type == "dt":
        clf = DecisionTreeClassifier(max_depth=opt_hp)
    else:
        clf = KNeighborsClassifier(n_neighbors=opt_hp)

    clf.fit(XLS, yLS)
    y_pred = clf.predict(XTS)
    test_score = accuracy_score(yTS, y_pred)

    return test_score, opt_hp


if __name__ == "__main__":
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/q4_plots'):
        os.mkdir(cwd + '/q4_plots')
    if not os.path.exists(cwd + '/q4_csv'):
        os.mkdir(cwd + '/q4_csv')

    dt_score_list2 = []
    knn_score_list2 = []
    dt_score_list1 = []
    knn_score_list1 = []

    # make the comparison for the dataset1
    dataset_list = []
    for i in range(5):
        random_seed = 29 + 10 * i
        dataset_list.append(make_dataset1(1500, random_seed))

    for dataset in dataset_list:
        hp_list = [1, 2, 4, 8, 10, 12, 14, 16]
        score_dt1, opt_dt_hp = run_cross_validation(dataset, "dt", hp_list)
        dt_score_list1.append(score_dt1)

        hp_list = [1, 5, 25, 125, 625]  # enlever 1200 sinon ça bug car LS < 1200 en 5fold cv
        score_knn1, opt_knn_hp = run_cross_validation(dataset, "knn", hp_list)
        knn_score_list1.append(score_knn1)

    # make the comparison for the dataset2
    dataset_list = []
    for i in range(5):
        random_seed = 29 + 10 * i
        dataset_list.append(make_dataset2(1500, random_seed))

    for dataset in dataset_list:
        hp_list = [1, 2, 4, 8, 10, 12, 14, 16]
        score_dt2, opt_dt_hp = run_cross_validation(dataset, "dt", hp_list)
        dt_score_list2.append(score_dt2)

        hp_list = [1, 5, 25, 125, 625]
        score_knn2, opt_knn_hp = run_cross_validation(dataset, "knn", hp_list)
        knn_score_list2.append(score_knn2)

    # write the result in a csv
    df = pd.DataFrame(
        {
            "clf_type": ("dt", "dt", "knn", "knn"),
            "dataset": ("dataset1", "dataset2", "dataset1", "dataset2"),
            "mean accuracy": (np.mean(dt_score_list1),
                              np.mean(dt_score_list2),
                              np.mean(knn_score_list1),
                              np.mean(knn_score_list2)),
            "std accuracy": (np.std(dt_score_list1),
                              np.std(dt_score_list2),
                              np.std(knn_score_list1),
                              np.std(knn_score_list2)),
        }
    )
    df.to_csv("q4_csv/" + "Q4.2_score" + ".csv")
    pass
