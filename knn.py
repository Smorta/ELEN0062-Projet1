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
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from plot import plot_boundary


# (Question 2)

def k_neigh(nb_neighbors, bool_plot, data):
    """Return train accuracy, test accuracy of
     K-nearest neighbors model trained on the dataset
    Parameters
    ----------
    nb_neighbors : int
            The numbers of the neighbors used by the model
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
    clf = KNeighborsClassifier(n_neighbors=nb_neighbors)
    clf.fit(XLS, yLS)

    y_fit = clf.predict(XLS)
    y_pred = clf.predict(XTS)

    # by definition of the accuracy
    LS_accuracy = np.sum(1 - np.absolute(yLS - y_fit)) / sizeLS
    TS_accuracy = np.sum(1 - np.absolute(yTS - y_pred)) / sizeTS

    if bool_plot:
        plot_boundary("knn_plots/knn" + str(nb_neighbors), clf, XTS, yTS, 0.1,
                      "Decision boundary for the \n number of neighbors of " + str(nb_neighbors))

    return LS_accuracy, TS_accuracy


if __name__ == "__main__":
    nb_neighbors = np.array([1, 5, 25, 125, 625, 1200])
    nb_hyper_parameters = np.size(nb_neighbors)
    # Q1 (a) simply the plot, retrieving LS_error and TS_error for Q1 (b)
    LS_scores = np.zeros(nb_hyper_parameters)
    TS_scores = np.zeros(nb_hyper_parameters)

    cwd = os.getcwd()
    if not os.path.exists(cwd + '/knn_plots'):
        os.mkdir(cwd + '/knn_plots')

    randomSeed = 19
    dataset = make_dataset2(1500, randomSeed)

    for i in range(nb_hyper_parameters):
        LS_scores[i], TS_scores[i] = k_neigh(nb_neighbors[i], True, dataset)
    # Q1 (b)
    plt.figure()
    plt.plot(nb_neighbors, 1 - LS_scores, '-o', nb_neighbors, 1 - TS_scores, '-s')
    plt.ylabel('Error [-]')
    plt.xlabel('Number of nearest neighbors [-]')
    plt.legend(['Error on learning set', 'Error on testing set'])
    plt.savefig('{}.pdf'.format("knn_plots/knn_error"), transparent=True)

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

    df = pd.DataFrame(
        {
            "Max Depth": nb_neighbors,
            "LS mean Error": ms_train_list,
            "LS std Error": std_train_list,
            "TS mean Error": ms_test_list,
            "TS std error": std_test_list
        }
    )
    df.to_csv("knn_score.csv")
    pass
