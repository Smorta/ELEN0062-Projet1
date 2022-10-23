import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from data import make_dataset2, make_dataset1
from sklearn.metrics import accuracy_score

def run_cross_validation(X, y, clf_type, hp_list, cv=5):
    if clf_type != "dt" and clf_type != "knn":
        raise Exception("the clf must be a 'dt' or 'knn'")
    if type(clf_type) is not str:
        raise Exception("the clf must be a string")

    mean_cv = np.zeros(len(hp_list))
    std_cv = np.zeros(len(hp_list))

    for i, hp in enumerate(hp_list):
        clf = None
        if clf_type == "dt":
            clf = DecisionTreeClassifier(max_depth=hp)
        else:
            clf = KNeighborsClassifier(n_neighbors=hp)

        score = cross_val_score(clf, X, y, cv=cv)
        mean_cv[i] = score.mean()
        std_cv[i] = score.std()

    opt_param_index = np.argmax(mean_cv)
    opt_hp = hp_list[opt_param_index]

    # finding the optimal parameter more precisely, focussing on the area of hp's near the current maximum
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
        new_mean_cv, new_std_cv, newopt_hp = run_cross_validation(X, y, clf_type, new_hp)
        opt_hp = newopt_hp

    return mean_cv, std_cv, opt_hp


def test_accuracy(opt_hp, clf_type):
    if clf_type != "dt" and clf_type != "knn":
        raise Exception("the clf must be a 'dt' or 'knn'")
    if type(clf_type) is not str:
        raise Exception("the clf must be a string")

    clf = None
    if clf_type == "dt":
        clf = DecisionTreeClassifier(max_depth=opt_hp)
    else:
        clf = KNeighborsClassifier(n_neighbors=opt_hp)

    sizeLS = 1200
    random_seed = 29
    dataset_list = []
    for i in range(5):
        dataset_list.append(make_dataset1(1500, random_seed + i * 10))
    score_list = []
    for dataset in dataset_list:
        X, y = dataset
        XLS = X[:sizeLS]
        yLS = y[:sizeLS]
        XTS = X[sizeLS:]
        yTS = y[sizeLS:]

        clf.fit(XLS, yLS)
        y_pred = clf.predict(XTS)  # on teste l'arbre sur le testing set

        score_list.append(accuracy_score(yTS, y_pred))

    return np.mean(score_list), np.std(score_list)


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


def save_cv_score_csv(hp_list, cv_mean_list, cv_std_list, filename):
    df = pd.DataFrame(
        {
            "HP_value": hp_list,
            "cv mean Error": cv_mean_list,
            "cv std Error": cv_std_list
        }
    )
    df.to_csv("q4_csv/" + filename + ".csv")


if __name__ == "__main__":
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/q4_plots'):
        os.mkdir(cwd + '/q4_plots')
    if not os.path.exists(cwd + '/q4_csv'):
        os.mkdir(cwd + '/q4_csv')

    # make the comparison for the dataset1

    random_seed = 29
    X, y = make_dataset1(1500, random_seed)

    hp_list = [1, 2, 4, 8, 10, 12, 14, 16]
    cv_mean, cv_std, opt_dt_hp1 = run_cross_validation(X, y, "dt", hp_list)
    print("optimal hyperparameter for dt (set1) is " + str(opt_dt_hp1))
    plot_cross_validation(cv_mean, cv_std, "Accuracy per tree depth",
                          "dt_cv_dataset1", "tree depth", "accuracy")
    save_cv_score_csv(hp_list, cv_mean, cv_std, "dt_dataset1")

    hp_list = [1, 5, 25, 125, 625]  # enlever 1200 sinon ça bug car LS < 1200 en 5fold cv
    cv_mean, cv_std, opt_knn_hp1 = run_cross_validation(X, y, "knn", hp_list)
    print("optimal hyperparameter for knn (set1) is " + str(opt_knn_hp1))
    plot_cross_validation(cv_mean, cv_std, "Accuracy per number of neighbors",
                          "knn_cv_dataset1", "number of neighbors", "accuracy")
    save_cv_score_csv(hp_list, cv_mean, cv_std, "knn_dataset1")

    # make the comparison for the dataset2
    random_seed = 29
    X, y = make_dataset2(1500, random_seed)

    hp_list = [1, 2, 4, 8, 10, 12, 14, 16]
    cv_mean, cv_std, opt_dt_hp2 = run_cross_validation(X, y, "dt", hp_list)
    print("optimal hyperparameter for dt (set2) is " + str(opt_dt_hp2))
    plot_cross_validation(cv_mean, cv_std, "Accuracy per tree depth",
                          "dt_cv_dataset2", "tree depth", "accuracy")
    save_cv_score_csv(hp_list, cv_mean, cv_std, "dt_dataset2")

    hp_list = [1, 5, 25, 125, 625]
    cv_mean, cv_std, opt_knn_hp2 = run_cross_validation(X, y, "knn", hp_list)
    print("optimal hyperparameter for knn (set2) is " + str(opt_knn_hp1))
    plot_cross_validation(cv_mean, cv_std, "Accuracy per number of neighbors",
                          "knn_cv_dataset2", "number of neighbors", "accuracy")
    save_cv_score_csv(hp_list, cv_mean, cv_std, "knn_dataset2")

    # test the accuracy of the optimal hyper_parameter over five generations on both dataset

    dt_score1_mean, dt_score1_std = test_accuracy(opt_dt_hp1, "dt")
    dt_score2_mean, dt_score2_std = test_accuracy(opt_dt_hp2, "dt")
    knn_score1_mean, knn_score1_std = test_accuracy(opt_knn_hp1, "dt")
    knn_score2_mean, knn_score2_std = test_accuracy(opt_knn_hp2, "dt")

    df = pd.DataFrame(
        {
            "clf_type": ("dt", "dt", "knn", "knn"),
            "dataset": ("dataset1", "dataset2", "dataset1", "dataset2"),
            "opt_hp": (opt_dt_hp1, opt_dt_hp2, opt_knn_hp1, opt_knn_hp2),
            "mean accuracy": (dt_score1_mean, dt_score2_mean, knn_score1_mean, knn_score2_mean),
            "std accuracy": (dt_score1_std, dt_score2_std, knn_score1_std, knn_score2_std)
        }
    )
    df.to_csv("q4_csv/" + "Q4.2_score" + ".csv")
    pass
