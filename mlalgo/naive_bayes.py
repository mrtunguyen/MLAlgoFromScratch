
import numpy as np


class NaiveBayesClassifier: 
    """Gaussian Naive Bayes classifier.
    """

    def fit(self, X, y):
        self._setup_input(X, y)
        self._init_params()
    
    def _setup_input(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = self.X.shape

    def _init_params(self):
        self._classes = np.unique(self.y)
        self._n_classes = len(self._classes)

        self._mean = np.zeros((self._n_classes, self.n_features))
        self._var = np.zeros((self._n_classes, self.n_features))

        self._class_prior = np.zeros(self._n_classes)

        for i, c in enumerate(self._classes):
            X_c = self.X[self.y == c]
            self._mean[i, :] = X_c.mean(axis=0)
            self._var[i, :] = X_c.var(axis=0)
            self._class_prior[i] = X_c.shape[0] / self.n_samples

    def predict(self, X):
        """Predict the class labels for the provided data"""
        return [self._predict(x) for x in X]
    
    def _predict(self, x):
        """Predict the class with the highest posterior probability"""
        posteriors = []

        for i, c in enumerate(self._classes):
            prior = np.log(self._class_prior[i])
            posterior = np.sum(np.log(self._pdf(i, x)))
            posterior = prior + posterior
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]
    
    def _pdf(self, class_idx, x):
        """Probability density function of the Gaussian distribution."""
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(- (x - mean) ** 2 / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator