from abc import abstractmethod
from collections import Counter
from typing import List, Union
import numpy as np


class KNN:
    """
    K Nearest Neighbors Algorithm

    Parameters:
    -----------
    k: int
        Number of neighbors to consider when making predictions

    """

    def __init__(self, k: int = 5) -> None:
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, X: np.ndarray) -> int:
        """Predict the class of a single sample
        Compute the euclidean distance between the sample and all points in the training set.
        Sort the distances and return the k nearest neighbors.
        Get the labels of the k nearest neighbors.
        """
        distances = [self._euclidean_distance(X, x_train) for x_train in self.X_train]
        k_indices = np.argsort(distances)[: self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        return self.aggregate(k_nearest_labels)

    @staticmethod
    def _euclidean_distance(x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    @abstractmethod
    def aggregate(self, labels: np.ndarray) -> int:
        return NotImplemented


class KNNClassifier(KNN):
    """
    K Nearest Neighbors Classifier

    """

    def aggregate(self, labels: Union[List, np.ndarray]) -> int:
        most_common = Counter(labels).most_common(1)
        return most_common[0][0]


class KNNRegressor(KNN):
    """
    K Nearest Neighbors Regressor

    """

    def aggregate(self, labels: Union[List, np.ndarray]) -> float:
        return np.mean(labels)

if __name__ == "__main__":
