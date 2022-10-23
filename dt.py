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


# Put your functions here
def DTC(max_dep, bool_plot, data):
    # max dep is the max_depth of the generated tree
    sizeOfSet = 1500
    sizeLS = 1200
    sizeTS = sizeOfSet - sizeLS
    X, y = data
    # question: ok de split ainsi ?
    XLS = X[:sizeLS]
    yLS = y[:sizeLS]
    XTS = X[sizeLS:]
    yTS = y[sizeLS:]

    # decision tree generation and training
    clf = DecisionTreeClassifier(max_depth=max_dep)
    clf.fit(XLS, yLS)

    y_fit = clf.predict(XLS)  # on teste l'arbre sur le learning set
    y_pred = clf.predict(XTS)  # on teste l'arbre sur le testing set

    # by definition of the errors
    LS_error = np.sum(np.absolute(yLS - y_fit)) / sizeLS
    TS_error = np.sum(np.absolute(yTS - y_pred)) / sizeTS

    if bool_plot:
        plot_boundary("dt_plots/dtFigure" + str(max_dep), clf, XTS, yTS, 0.1,
                      "Decision boundary for the \n max depth value of " + str(max_dep))

    return LS_error, TS_error, clf.get_depth()


def plot_mean_accuracy(cv_mean, cv_std, hp_list, title, filename, y_label, x_label):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    ax.plot(hp_list, cv_mean, '-o', label='mean cross-validation accuracy', alpha=0.9)
    ax.fill_between(hp_list, cv_mean - 2 * cv_std, cv_mean + 2 * cv_std, alpha=0.2)
    ylim = plt.ylim()
    ax.set_title(title, fontsize=16)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.set_ylim(ylim)
    ax.set_xticks(hp_list)
    ax.legend(loc="lower right")
    plt.savefig(filename + ".pdf", transparent=True)


if __name__ == "__main__":
    max_depths_in = np.array([1, 2, 4, 8, None])
    nbDepths = np.size(max_depths_in)
    # Q1 (a) simply the plot, retrieving LS_error and TS_error for Q1 (b)
    LS_errors = np.zeros(nbDepths)
    TS_errors = np.zeros(nbDepths)
    max_depths_out = np.zeros(nbDepths)  # va etre pareil que in sauf pr None on aura une valeur finie

    cwd = os.getcwd()
    if not os.path.exists(cwd + '/dt_plots'):
        os.mkdir(cwd + '/dt_plots')

    randomSeed = 19
    dataset = make_dataset2(1500, randomSeed)

    for i in range(nbDepths):
        LS_errors[i], TS_errors[i], max_depths_out[i] = DTC(max_depths_in[i], True, dataset)
    # Q1 (b)
    plt.figure()
    plt.plot(max_depths_out, LS_errors, '-o', max_depths_out, TS_errors, '-s')
    plt.ylabel('Error')
    plt.xlabel('Effective depth of the decision tree')
    plt.legend(['Error on learning sample', 'Error on testing sample'])
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

    plot_mean_accuracy(np.array(ms_test_list), np.array(std_test_list), max_depths_out, "",
                       "dt_plots/mean_error", "error", "max_depth")

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
