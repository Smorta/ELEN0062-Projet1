"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
from data import make_dataset2
from sklearn.tree import DecisionTreeClassifier
from plot import plot_boundary


def DTC(max_dep, bool_plot, data):
    """Return train accuracy, test accuracy and the depth of
     the decision tree trained on the dataset
    Parameters
    ----------
    max_dep : int
            The maximal depth of the decision tree
    bool_plot : boolean
            A flag for activate the plot of the decision boundaries of the model
    data : array-like, shape = [n_samples, n_features + 1]
            The dataset
    Returns
    -------
    LS_accuracy : float
            The normalized train accuracy of the model on the dataset
    TS_accuracy : float
            The normalized test accuracy of the model on the dataset
    clf.get_depth() : int
            The depth of the trained decision tree
    """

    sizeOfSet = 1500
    sizeLS = 1200
    sizeTS = sizeOfSet - sizeLS
    X, y = data
    # splitting the data set into learning set and testing set
    XLS = X[:sizeLS]
    yLS = y[:sizeLS]
    XTS = X[sizeLS:]
    yTS = y[sizeLS:]

    # decision tree generation and training
    clf = DecisionTreeClassifier(max_depth=max_dep)
    clf.fit(XLS, yLS)

    y_fit = clf.predict(XLS)  # on teste l'arbre sur le learning set
    y_pred = clf.predict(XTS)  # on teste l'arbre sur le testing set

    # by definition of the accuracy
    LS_accuracy = np.sum(1 - np.absolute(yLS - y_fit)) / sizeLS
    TS_accuracy = np.sum(1 - np.absolute(yTS - y_pred)) / sizeTS

    if bool_plot:
        plot_boundary("dt_plots/dt" + str(max_dep), clf, XTS, yTS, 0.1,
                      "Decision boundary for the \n max depth value of " + str(max_dep))

    return LS_accuracy, TS_accuracy, clf.get_depth()


if __name__ == "__main__":
    max_depths_in = np.array([1, 2, 4, 8, None])
    nbDepths = np.size(max_depths_in)
    # Q1 (a) simply the plot, retrieving LS_error and TS_error for Q1 (b)
    LS_score = np.zeros(nbDepths)
    TS_score = np.zeros(nbDepths)
    # will be the same as max_depths_in expected for None that will the fitted max_depth
    max_depths_out = np.zeros(nbDepths)

    cwd = os.getcwd()
    if not os.path.exists(cwd + '/dt_plots'):
        os.mkdir(cwd + '/dt_plots')

    randomSeed = 19
    dataset = make_dataset2(1500, randomSeed)

    for i in range(nbDepths):
        LS_score[i], TS_score[i], max_depths_out[i] = DTC(max_depths_in[i], True, dataset)
    # Q1 (b)
    plt.figure()
    plt.plot(max_depths_out, 1 - LS_score, '-o', max_depths_out, 1 - TS_score, '-s')
    plt.ylabel('Error [-]')
    plt.xlabel('Effective depth of the decision tree [-]')
    plt.legend(['Error on learning set', 'Error on testing set'])
    plt.savefig('{}.pdf'.format("dt_plots/dt_error"), transparent=True)

    # Q2
    nbGenerations = 5
    LS_score_list = [[] for i in range(nbDepths)]
    TS_score_list = [[] for i in range(nbDepths)]

    for i in range(nbGenerations):
        score_dic = {}
        randomSeed = 19 + 10 * i
        dataset = make_dataset2(1500, randomSeed)
        for j, depth in enumerate(max_depths_in):
            LS_error_tmp, TS_errors_tmp, curr_depth = DTC(depth, False, dataset)
            LS_score_list[j].append(LS_error_tmp)
            TS_score_list[j].append(TS_errors_tmp)

    std_train_list = []
    std_test_list = []
    ms_train_list = []
    ms_test_list = []

    for i in range(nbDepths):
        std_train_list.append(np.std(LS_score_list[i]))
        ms_train_list.append(np.mean(LS_score_list[i]))
        std_test_list.append(np.std(TS_score_list[i]))
        ms_test_list.append(np.mean(TS_score_list[i]))

    df = pd.DataFrame(
        {
            "Max Depth": max_depths_in,
            "LS mean Error": ms_train_list,
            "LS std Error": std_train_list,
            "TS mean Error": ms_test_list,
            "TS std error": std_test_list
        }
    )
    df.to_csv("dt_score.csv")
    pass
