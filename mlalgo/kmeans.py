import random
from typing import List
import numpy as np

class Kmeans:
    """Kmeans clustering algorithm implementation from scratch

    Partition data into k clusters, where each observation belongs to the cluster with the nearest mean,
    serving as a prototype of the cluster. Finds clusters by repeatedly assigning each data point to the
    cluster with the nearest centroid and iterating until the assignments converge (meaning they don't
    change during an iteration) or the maximum number of iterations is reached.

    Parameters:
    -----------
    k: int
        Number of clusters to partition data into
    max_iter: int
        Maximum number of iterations to run the algorithm for
    init: str
        Method for initializing centroids. Must be one of 'random' or 'kmeans++'
        'random' - randomly choose k data points from the dataset to be the initial centroids
        'kmeans++' - choose the first centroid randomly, then choose the next centroid from the remaining
            data points with probability proportional to the distance from the point to the nearest centroid
            In objective to create larger distances between initial clusters to improve the algorithm's convergence
            and avoid degenerate cases where a centroid ends up with no points assigned to it
    """

    def __init__(self, k: int = 5, max_iters: int = 100, init="random") -> None:
        self.k = k
        self.max_iters = max_iters
        self.init = init
        self.centroids = []
        self.clusters = [[] for _ in range(self.k)]

    def _setup(self, X: np.ndarray, y=None) -> None:
        """Initialize input"""
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        self.X = X
        self.n_samples, self.n_features = X.shape

    def _choose_next_centroid(self) -> np.ndarray:
        """Choose the next centroid from the remaining data points with probability proportional
        to the distance from the point to the nearest centroid"""

        distances = np.array(
            [min([np.linalg.norm(x - c) for c in self.centroids]) for x in self.X]
        )
        squared_distances = distances**2
        probs = squared_distances / squared_distances.sum()
        index = np.random.choice(range(self.n_samples), p=probs)
        return self.X[index]

    def _initialize_centroids(self, init) -> None:
        """Initialize centroids by choosing k random points from the dataset"""

        if init == "random":
            self.centroids = [
                self.X[i]
                for i in np.random.choice(range(self.n_samples), self.k, replace=False)
            ]
        elif init == "kmeans++":
            self.centroids = [random.choice(self.X)]
            while len(self.centroids) < self.k:
                self.centroids.append(self._choose_next_centroid())
        else:
            raise ValueError("init must be one of 'random' or 'kmeans++'")

    def fit(self, X: np.ndarray, y=None) -> None:
        self._setup(X, y)
        self._initialize_centroids(self.init)
        
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)
            old_centroids = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            if self._is_converged(old_centroids, self.centroids):
                break
    
    def predict(self, X: np.ndarray) -> List[int]:
        """Predict which cluster each sample belongs to"""
        return self._get_cluster_labels(X, self.centroids)
    
    def _create_clusters(self, centroids: np.ndarray) -> List[List[np.ndarray]]:
        """Assign samples to closest centroids to create clusters"""
        clusters = [[] for _ in range(self.k)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def _closest_centroid(self, sample: np.ndarray, centroids: np.ndarray) -> int:
        """Return index of the closest centroid to a sample"""
        distances = [np.linalg.norm(sample - point) for point in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx # type: ignore
    
    def _get_centroids(self, clusters: List[List[np.ndarray]]) -> np.ndarray:
        """Calculate new centroids as the mean of the samples in each cluster"""
        centroids = np.zeros((self.k, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids
    
    def _is_converged(self, old_centroids: np.ndarray, centroids: np.ndarray) -> bool:
        """Check if the centroids have changed since the last iteration"""
        distances = [np.linalg.norm(old_centroids[i] - centroids[i]) for i in range(self.k)]
        return sum(distances) == 0 # type: ignore
    
    def _get_cluster_labels(self, X: np.ndarray, centroids: np.ndarray) -> List[int]:
        """Return the index of the cluster each sample belongs to"""
        return [self._closest_centroid(sample, centroids) for sample in X]
    
if __name__ == "__main__":
    X = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    kmeans = Kmeans(k=2, max_iters=100, init="random")
    kmeans.fit(X)
    print(kmeans.predict(X))
    print(kmeans.centroids)
