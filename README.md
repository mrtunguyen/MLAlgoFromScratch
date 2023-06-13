# MLAlgoFromScratch
Implementation of ML algorithms from scratch in Python

## Install environment 
```python 
python -m venv .venv
pip install poetry 
poetry install
```

## Algorithms
### KMmeans
**Kmeans** algorithm is an iterative algorithm that tries to partition the dataset into *K* predefined distinct non-overlapping subgroups (clusters) where each data point belongs to **only one group**. It tries to make the intra-cluster data points as similar as possible while also keeping the clusters as different (far) as possible. 

It assigns data points to a cluster such that the sum of the squared distance between the data points and the cluster’s centroid (arithmetic mean of all the data points that belong to that cluster) is at the minimum. 

The less variation we have within clusters, the more homogeneous (similar) the data points are within the same cluster.
### KNN
 It works by finding the distances between a query and all the examples in the data, selecting the specified number of examples (K) closest to the query, and then voting for the most frequent label (in the case of classification) or averaging the labels (in the case of regression)

 In the classification problem, KNN assigns a class label based on a majority vote—i.e., the label that is most frequently represented around a given data point is used. For regression problems, the algorithm takes the average of the K nearest neighbors to make a prediction.

 Some advantages of the KNN algorithm include:

- Easy to implement
- Adapts easily to new training samples
- Few hyperparameters required (K and distance metric)

However, there are also some disadvantages:

- Does not scale well with large datasets
- Suffers from the curse of dimensionality, i.e., it doesn't perform well with high-dimensional data inputs


### Gaussian Mixture

A Gaussian Mixture Model (GMM) is a probabilistic model used in machine learning to represent a dataset as a collection of Gaussian distributions, also known as components or clusters. GMM assumes that the dataset is generated from a mixture of these Gaussian distributions, where each data point is associated with a certain component with a corresponding probability.

How a GMM works?

- Initialization: Initially, you need to determine the number of components, K, in the GMM. You can either set this number based on prior knowledge or use techniques like the Bayesian Information Criterion (BIC) or cross-validation to estimate the optimal number of components.

- Parameter Initialization: Randomly initialize the parameters of the GMM. These parameters include the mean vectors, covariance matrices, and mixture weights for each component.

- Expectation-Maximization (EM) Algorithm:
    -  E-Step (Expectation): Calculate the probability or responsibility of each data point belonging to each component. This is done by applying Bayes' theorem and using the current parameter estimates.
    - M-Step (Maximization): Update the parameters of the GMM by maximizing the expected log-likelihood obtained in the E-step. Specifically, update the mean vectors, covariance matrices, and mixture weights based on the weighted data points.

- Iterative Refinement: Repeat the E-step and M-step until convergence is reached. Convergence is typically determined by a threshold or when the change in the log-likelihood falls below a certain value.

- Model Evaluation: Once the GMM has converged, you can evaluate the quality of the model. This may involve assessing the log-likelihood of the dataset, comparing different GMMs with different numbers of components using information criteria, or using other evaluation metrics such as silhouette score or clustering accuracy.

- Cluster Assignment: After training, you can assign each data point to the most likely component based on the calculated responsibilities. This allows you to identify the cluster or component to which each data point belongs.

- Prediction: Given a new data point, you can use the trained GMM to estimate its likelihood of belonging to each component. This can be used for tasks like anomaly detection or generating new samples from the learned distributions.

### Factorization Machine
Factorization Machines (FM) are a supervised machine learning model designed to capture interactions between features in a dataset. They are particularly useful for handling high-dimensional and sparse data, where feature interactions play a crucial role in making accurate predictions.

At a high level, FM models the interactions between features by factorizing them into a lower-dimensional latent space. It assumes that each feature can be represented by a set of latent factors, and the interactions between features can be captured through the dot product of their corresponding latent factor vectors.

