"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-
import os

import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from data import make_dataset1, make_dataset2
from sklearn.tree import DecisionTreeClassifier, plot_tree
from plot import plot_boundary


def test_accuracy(clf, X_train, X_test, y_train, y_test):
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    train_score = accuracy_score(y_train, y_train_pred)
    test_score = accuracy_score(y_test, y_test_pred)
    return train_score, test_score


if __name__ == "__main__":

    # question 1
    X, y = make_dataset2(1500, 17)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    clf_array = []
    depth_array = [1, 2, 4, 8, None]

    for depth in depth_array:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
        clf.fit(X_train, y_train)
        clf_array.append(clf)

    # plot boundaries of the classifiers for each depth
    cwd = os.getcwd()
    if not os.path.exists(cwd + '/plots'):
        os.mkdir(cwd + '/plots')

    for i, clf in enumerate(clf_array):
        fname = "plots/dt_bnd" + str(depth_array[i])
        title = "max_depth = " + str(clf.get_depth())
        plot_boundary(fname, clf, X_test, y_test, title=title)
        plot_tree(clf, filled=True)
        plt.savefig("plots/dt_tree" + str(depth_array[i]) + ".png", format="png")

    # question 2
    seeds_list = [17, 22, 45, 66, 52]
    train_score_list = {}
    test_score_list = {}
    depth_array = [1, 2, 4, 8, None]

    for seed in seeds_list:
        X, y = make_dataset2(1500, seed)
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

        score_dic = {}
        for i, depth in enumerate(depth_array):
            clf = DecisionTreeClassifier(max_depth=depth, random_state=0)
            clf.fit(X_train, y_train)
            if depth is None:
                depth = clf.get_depth()
            train_score, test_score = test_accuracy(clf, X_train, X_test, y_train, y_test)
            if depth not in train_score_list.keys():
                train_score_list[depth] = []
                test_score_list[depth] = []
            train_score_list[depth].append(train_score)
            test_score_list[depth].append(test_score)

    std_train_list = []
    std_test_list = []
    ms_train_list = []
    ms_test_list = []

    for depth in train_score_list.keys():
        std_train_list.append(np.std(train_score_list[depth]))
        ms_train_list.append(np.mean(train_score_list[depth]))
        std_test_list.append(np.std(test_score_list[depth]))
        ms_test_list.append(np.mean(test_score_list[depth]))

    plt.figure()
    plt.errorbar(
        train_score_list.keys(),
        ms_train_list,
        std_train_list,
        fmt='o',
        label='train accuracy',
        linewidth=2,
        capsize=6
    )
    plt.errorbar(
        train_score_list.keys(),
        ms_test_list,
        std_test_list,
        fmt='o',
        label='test accuracy',
        linewidth=2,
        capsize=6
    )
    plt.xlabel("max_depth")
    plt.ylabel("accuracy")
    plt.legend()
    plt.show()
