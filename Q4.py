import numpy as np
from matplotlib import pyplot as plt
import os
from knn import k_neigh
from dt import DTC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from data import make_dataset2, make_dataset1


def run_cross_validation(datasets, clf_type, hp_list, cv=5):
    if clf_type != "dt" and clf_type != "knn":
        raise Exception("the clf must be a 'dt' or 'knn'")
    if type(clf_type) is not str:
        raise Exception("the clf must be an string")

    cv_list = [[] for _ in range(len(hp_list))]
    for i, hp in enumerate(hp_list):
        clf = None
        if clf_type == "dt":
            clf = DecisionTreeClassifier(max_depth=hp)
        else:
            clf = KNeighborsClassifier(n_neighbors=hp)

        for dataset in datasets:
            X, y = dataset
            score = cross_val_score(clf, X, y, cv=cv)
            cv_list[i].append(score.mean())

    return cv_list, np.mean(cv_list, axis=1), np.std(cv_list, axis=1)


def plot_cross_validation(cv_mean_list, cv_std_list, title, filename, x_label, y_label):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(hp_list, cv_mean_list, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(hp_list, cv_mean_list - 2 * cv_std_list, cv_mean_list + 2 * cv_std_list, alpha=0.2)
    ylim = plt.ylim()
    ax.set_title('tuning of hyperParameter', fontsize=16)
    ax.set_xlabel('Tree depth', fontsize=14)
    ax.set_ylabel('Accuracy', fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(hp_list)
    ax.legend(loc="lower right")
    plt.savefig("q4_plots/" + filename + ".pdf", transparent=True)


if __name__ == "__main__":
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/q4_plots'):
        os.mkdir(cwd + '/q4_plots')

    dataset_list = []
    for i in range(5):
        random_seed = 29+10*i
        dataset_list.append(make_dataset1(1500, random_seed))
    hp_list = range(1, 20)
    cv_list, cv_mean_list, cv_std_list = run_cross_validation(dataset_list, "dt", hp_list)
    plot_cross_validation(cv_mean_list, cv_std_list, "Accuracy per tree depth",
                          "dt_cv_dataset1", "tree depth", "accuracy")
    hp_list = range(1, 50)
    cv_list, cv_mean_list, cv_std_list = run_cross_validation(dataset_list, "knn", hp_list)
    plot_cross_validation(cv_mean_list, cv_std_list, "Accuracy per number of neighbors",
                          "knn_cv_dataset1", "number of neighbors", "accuracy")
    dataset_list = []
    for i in range(5):
        random_seed = 29 + 10 * i
        dataset_list.append(make_dataset2(1500, random_seed))
    hp_list = range(1, 20)
    cv_list, cv_mean_list, cv_std_list = run_cross_validation(dataset_list, "dt", hp_list)
    plot_cross_validation(cv_mean_list, cv_std_list, "Accuracy per tree depth",
                          "dt_cv_dataset2", "tree depth", "accuracy")
    hp_list = range(1, 50)
    cv_list, cv_mean_list, cv_std_list = run_cross_validation(dataset_list, "knn", hp_list)
    plot_cross_validation(cv_mean_list, cv_std_list, "Accuracy per number of neighbors",
                          "knn_cv_dataset2", "number of neighbors", "accuracy")

    pass
