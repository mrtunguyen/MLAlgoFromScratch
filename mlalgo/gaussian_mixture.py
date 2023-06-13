# coding: utf-8

from operator import is_
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
        self.likelihoods = self._get_likelihood(self.X)
        self.assigments = self.likelihoods.argmax(axis=1)
        self.likelihoods /= self.likelihoods.sum(axis=1, keepdims=True)

    def _m_step(self):
        """Update the mean and covariance matrix based on the probability computed in E-step"""
        for i in range(self.n_components):
            weight = self.likelihoods[:, i].sum()
            self.weights[i] = weight / self.n_samples
            self.mean[i] = self.likelihoods[:, i].dot(self.X) / weight
            self.cov[i] = (
                self.likelihoods[:, i]
                * (self.X - self.mean[i]).T.dot(self.X - self.mean[i])
                / weight
            )

    def _is_converged(self):
        """Check if the total likelihood of the dataset does not change much"""
        if not hasattr(self, "prev_likelihood"):
            self.prev_likelihood = 0
            return False
        diff = np.abs(self.likelihoods.sum() - self.prev_likelihood)
        self.prev_likelihood = self.likelihoods.sum()
        return diff <= self.tol

    def _get_likelihood(self, X):
        """Compute the likelihood of the dataset"""
        likelihood = np.zeros((X.shape[0], self.n_components))
        for i in range(self.n_components):
            likelihood[:, i] = self.weights[i] * multivariate_normal.pdf(
                X, self.mean[i], self.cov[i]
            )
        return likelihood