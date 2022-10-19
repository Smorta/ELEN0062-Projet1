"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
#! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import os
from data import make_dataset2
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary


# (Question 2)

def k_neigh(nb_neighbors, bool_plot, data):
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
    clf = KNeighborsClassifier(n_neighbors=nb_neighbors)
    clf.fit(XLS, yLS)

    y_fit = clf.predict(XLS)  # on teste l'arbre sur le learning set
    y_pred = clf.predict(XTS)  # on teste l'arbre sur le testing set

    # by definition of the errors
    LS_error = np.sum(np.absolute(yLS - y_fit)) / sizeLS
    TS_error = np.sum(np.absolute(yTS - y_pred)) / sizeTS

    if bool_plot:
        plot_boundary("knn_plots/knnFigure" + str(nb_neighbors), clf, XTS, yTS, 0.1,
                      "Decision boundary for the \n number of neighbors of " + str(nb_neighbors))

    return LS_error, TS_error


if __name__ == "__main__":
    nb_neighbors = np.array([1, 5, 25, 125, 625, 1200])
    nb_hyper_parameters = np.size(nb_neighbors)
    # Q1 (a) simply the plot, retrieving LS_error and TS_error for Q1 (b)
    LS_errors = np.zeros(nb_hyper_parameters)
    TS_errors = np.zeros(nb_hyper_parameters)

    cwd = os.getcwd()
    if not os.path.exists(cwd + '/knn_plots'):
        os.mkdir(cwd + '/knn_plots')

    randomSeed = 19
    dataset = make_dataset2(1500, randomSeed)

    for i in range(nb_hyper_parameters):
        LS_errors[i], TS_errors[i] = k_neigh(nb_neighbors[i], True, dataset)
    # Q1 (b)
    plt.figure()
    plt.plot(nb_neighbors, LS_errors, '-o', nb_neighbors, TS_errors, '-s')
    plt.ylabel('Error')
    plt.xlabel('Effective depth of the decision tree')
    plt.legend(['Error on learning sample', 'Error on testing sample'])
    plt.savefig('{}.pdf'.format("knn_plots/dt_error"), transparent=True)

    # Q2
    nbGenerations = 5
    LS_score_list = [[] for i in range(nb_hyper_parameters)]
    TS_score_list = [[] for i in range(nb_hyper_parameters)]

    for i in range(nbGenerations):
        score_dic = {}
        randomSeed = 19 + 10 * i
        dataset = make_dataset2(1500, randomSeed)
        for j in range(nb_hyper_parameters):
            LS_error_tmp, TS_errors_tmp = k_neigh(nb_neighbors[j], False, dataset)
            LS_score_list[j].append(LS_error_tmp)
            TS_score_list[j].append(TS_errors_tmp)

    std_train_list = []
    std_test_list = []
    ms_train_list = []
    ms_test_list = []

    for i in range(nb_hyper_parameters):
        std_train_list.append(np.std(LS_score_list[i]))
        ms_train_list.append(np.mean(LS_score_list[i]))
        std_test_list.append(np.std(TS_score_list[i]))
        ms_test_list.append(np.mean(TS_score_list[i]))

    plt.figure()
    plt.errorbar(nb_neighbors, ms_train_list, std_train_list, fmt='o',
                 label='train error', linewidth=2, capsize=6)
    plt.errorbar(nb_neighbors, ms_test_list, std_test_list, fmt='o',
                 label='test error', linewidth=2, capsize=6)
    plt.xlabel("max_depth")
    plt.ylabel("error")
    plt.legend()
    plt.savefig('{}.pdf'.format("knn_plots/dt_mean_error"), transparent=True)
    pass
