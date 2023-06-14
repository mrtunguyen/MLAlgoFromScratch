# coding: utf-8

from ast import Assign
from operator import is_
from urllib import response
from matplotlib import pyplot as plt
import numpy as np
from kmeans import Kmeans
from scipy.stats import multivariate_normal


class GaussianMixture:
    """Gaussian Mixture Model

    Find clusters by repeatedly performing Expectation-Maximization (EM) algorithm on the dataset.
    Gaussian Mixture Model (GMM) assumes the datasets is distributed in multivariate Gaussian distributions,
    and the goal is to find the parameters of these distributions, i.e. mean and covariance matrix.
    E-step: compute the probability of each data point belonging to each distribution, given the mean and covariance.
    M-step: update the mean and covariance based on the probability computed in E-step.
    Repeat E-step and M-step until convergence (the total likelihood of the dataset does not change much, i.e less than the tolerance).

    Parameters
    ----------
    n_components : int, default=1
        Number of components/clusters
    max_iter : int, default=100
        Maximum number of iterations to train the algorithm
    tol : float, default=1e-3
        Tolerance for convergence
    init: str, default='random'
        Method to initialize the mean and covariance matrix of each component.
        'random' - randomly initialize the mean and covariance matrix
        'kmeans' - use KMeans to initialize the mean and covariance matrix
    """

    def __init__(self, n_components=1, max_iter=100, tol=1e-3, init="random"):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.init = init
        self.likelihood = []
        self.assignments = None
        self.reponsibilities = None

    def _setup_input(self, X):
        """Check the input data and convert to numpy array"""
        self.X = X
        self.n_samples, self.n_features = self.X.shape

    def fit(self, X):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features

        Returns
        -------
        self : object
            Returns self
        """
        self._setup_input(X)
        self._init_components()

        for _ in range(self.max_iter):
            self._e_step()
            self._m_step()
            if self._is_converged():
                break

        return self

    def _init_components(self):
        """Initialize the mean and covariance matrix of each component"""
        self.weights = np.ones(self.n_components)
        if self.init == "random":
            self.mean = np.random.rand(self.n_components, self.n_features)
            self.cov = np.array(
                [np.eye(self.n_features) for _ in range(self.n_components)]
            )
        elif self.init == "kmeans":
            kmeans = Kmeans(k=self.n_components, init="++")
            kmeans.fit(self.X)
            self.assigments = kmeans.predict(self.X)
            self.mean = kmeans.centroids
            self.cov = []
            for i in np.unique(self.assigments):
                cluster = self.X[self.assigments == i]
                self.cov.append(np.cov(cluster.T))
                self.weights[int(i)] = (self.assigments == i).sum()
        else:
            raise ValueError("init must be either 'random' or 'kmeans'")
        self.weights /= self.weights.sum()

    def _e_step(self):
        """Compute the probability of each data point belonging to each distribution"""
        likelihoods = self._get_likelihood(self.X)
        self.likelihood.append(likelihoods.sum())
        weighted_likelihoods = likelihoods * self.weights
        self.assigments = weighted_likelihoods.argmax(axis=1)
        self.reponsibilities /= weighted_likelihoods.sum(axis=1)[: np.newaxis]

    def _m_step(self):
        """Update the mean and covariance matrix based on the probability computed in E-step"""
        weights = self.reponsibilities.sum(axis=0)
        for assignment in range(self.n_components):
            responsibity = self.reponsibilities[:, assignment][:, np.newaxis]
            self.mean[assignment] = (responsibity * self.X).sum(
                axis=0
            ) / responsibity.sum()
            self.cov[assignment] = (
                (responsibity * (self.X - self.mean[assignment])).T
                @ ((self.X - self.mean[assignment]) * responsibity)
                / weights[assignment]
            )
        self.weights = weights / weights.sum()

    def _is_converged(self):
        """Check if the total likelihood of the dataset does not change much"""
        if (len(self.likelihood) > 1) and (self.likelihood[-1] - self.likelihood[-2] <= self.tol):
            return True
        return False

    def _get_likelihood(self, X):
        """Compute the likelihood of the dataset"""
        likelihood = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            likelihood[:, i] = multivariate_normal.pdf(
                X, self.mean[i], self.cov[i]
            )
        return likelihood
    
    def _predict(self, X):
        """Predict the cluster of each data point"""
        if not X.shape: 
            return self.assignments
        likelihood = self._get_likelihood(X)
        weighted_likelihood = likelihood * self.weights
        assignments = weighted_likelihood.argmax(axis=1)
        return assignments
    
    def predict(self, X):
        """Predict the cluster of each data point"""
        return self._predict(X)
    
    def plot(self, data=None, ax=None, holdon=False):
        """Plot contour for 2D data."""
        if not (len(self.X.shape) == 2 and self.X.shape[1] == 2):
            raise AttributeError("Only support for visualizing 2D data.")

        if ax is None:
            _, ax = plt.subplots()

        if data is None:
            data = self.X
            assignments = self.assignments
        else:
            assignments = self.predict(data)

        COLOR = "bgrcmyk"
        cmap = lambda assignment: COLOR[int(assignment) % len(COLOR)]

        # generate grid
        delta = 0.025
        margin = 0.2
        xmax, ymax = self.X.max(axis=0) + margin
        xmin, ymin = self.X.min(axis=0) - margin
        axis_X, axis_Y = np.meshgrid(np.arange(xmin, xmax, delta), np.arange(ymin, ymax, delta))

        def grid_gaussian_pdf(mean, cov):
            grid_array = np.array(list(zip(axis_X.flatten(), axis_Y.flatten())))
            return multivariate_normal.pdf(grid_array, mean, cov).reshape(axis_X.shape)

        # plot scatters
        if assignments is None:
            c = None
        else:
            c = [cmap(assignment) for assignment in assignments]
        ax.scatter(data[:, 0], data[:, 1], c=c)

        # plot contours
        for assignment in range(self.n_components):
            ax.contour(
                axis_X,
                axis_Y,
                grid_gaussian_pdf(self.mean[assignment], self.cov[assignment]),
                colors=cmap(assignment),
            )

        if not holdon:
            plt.show()