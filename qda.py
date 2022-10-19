"""
University of Liege
ELEN0062 - Introduction to machine learning
Project 1 - Classification algorithms
"""
# ! /usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import math

from sklearn.base import BaseEstimator, ClassifierMixin


class QuadraticDiscriminantAnalysis(BaseEstimator, ClassifierMixin):

    def __init__(self):
        self.pi_ = np.array([0.0, 0.0])

        # Array of two points
        self.mean_ = np.array([[0.0, 0.0], [0.0, 0.0]])

        # in binary classification case, will always be a 2x2 matrix
        self.cov_0 = np.array([[0.0, 0.0], [0.0, 0.0]])
        self.cov_1 = np.array([[0.0, 0.0], [0.0, 0.0]])

    def fit(self, X, y, lda=False):

        """Fit a linear discriminant analysis model using the training set
        (X, y).

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.

        Returns
        -------
        self : object
            Returns self.
        """
        # Input validation
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2 dimensional")

        y = np.asarray(y)
        if y.shape[0] != X.shape[0]:
            raise ValueError("The number of samples differs between X and y")

        self.lda = lda

        X_y0 = X[y == 0]
        X_y1 = X[y == 1]

        # Estimate pi ( = number of points of a class divided by the number of points)
        # We could also have assumed equiprobable classes
        self.pi_[0] = X_y0.shape[0] / y.shape[0]
        self.pi_[1] = 1 - self.pi_[0]

        # The maximum likelihood estimator of the expectation µ of a normal distribution
        self.mean_[0] = [np.mean(X_y0[:, 0]), np.mean(X_y0[:, 1])]
        self.mean_[1] = [np.mean(X_y1[:, 0]), np.mean(X_y1[:, 1])]

        if lda:
            self.cov_0 = np.cov(np.transpose(X_y0))
            self.cov_1 = np.cov(np.transpose(X_y0))
        else:
            self.cov_0 = np.cov(np.transpose(X_y0))
            self.cov_1 = np.cov(np.transpose(X_y1))

        return self

    def predict(self, X):
        """Predict class for X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape = [n_samples]
            The predicted classes, or the predict values.
        """
        return np.array([0 if p > 0.5 else 1
                         for p in self.predict_proba(X)[:, 1]])

    def predict_proba(self, X):
        """Return probability estimates for the test data X.

        Parameters
        ----------
        X : array-like of shape = [n_samples, n_features]
            The input samples.

        Returns
        -------
        p : array of shape = [n_samples, n_classes]
            The class probabilities of the input samples. Classes are ordered
            by lexicographic order.
        """
        f_0 = np.zeros(len(X))
        f_1 = np.zeros(len(X))

        for i in range(len(X)):
            f_0[i] = (1 / (math.sqrt(2 * math.pi * np.linalg.det(self.cov_0)))
                      * math.exp(- np.dot(np.dot((X[i] - self.mean_[0]), np.linalg.inv(self.cov_0)),
                                          np.transpose(X[i] - self.mean_[0])))
                      / 2)

            f_1[i] = (1 / (math.sqrt(2 * math.pi * np.linalg.det(self.cov_1)))
                      * math.exp(- np.dot(np.dot((X[i] - self.mean_[1]), np.linalg.inv(self.cov_1)),
                                          np.transpose(X[i] - self.mean_[1])))
                      / 2)

        p = f_0 * self.pi_[0] / (
                (f_0 * self.pi_[0]) + (f_1 * self.pi_[1]))
        _p = 1 - p
        n_samples = len(X)

        return np.hstack((p.reshape(n_samples, 1), _p.reshape(n_samples, 1)))


if __name__ == "__main__":
    from data import make_dataset1, make_dataset2
    from plot import plot_boundary
    import matplotlib.pyplot as plt

    n_points = 1500
    for make_set, fname in ((make_dataset1, "dataset1"),
                            (make_dataset2, "dataset2")):
        X, y = make_set(n_points)

        qdl = QuadraticDiscriminantAnalysis()
        qdl.fit(X[0:1200], y[0:1200], False)
        plot_boundary(fname, qdl, X[1200:1500], y[1200:1500], title="QDL")
