from abc import abstractmethod
from xml.dom.minidom import Element
import numpy as np
from tqdm import tqdm
from autograd import elementwise_grad

def mean_squared_error(y_true, y_pred):
    return np.mean(np.power(y_true - y_pred, 2))

def binary_crossentropy(y_true, y_pred):
    return np.mean(
        -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    )

class FMBase:
    """
    Base class for Factorization Machine
    Reference: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf

    Factorization Machine is a supervised learning algorithm that can be used for both classification and regression tasks.
    It is a generalization of linear regression that models all possible interactions between features using factorized parameters.
    FM is able to estimate interactions even in problems with huge sparsity in the data (e.g. recommender systems) and is 
    therefore a popular choice for these types of problems. 

    Parameters
    ----------
    n_iter : int, default=100
        Number of iterations to train the algorithm
    n_factors : int, default=10
        Number/dimension of factors
    learning_rate : float, default=0.01
        Learning rate for gradient descent
    reg_w : float, default=0.01
        Regularization parameter for linear weights
    reg_v : float, default=0.01
        Regularization parameter for factorized weights
    random_state : int, default=None
        Seed for the random number generator
    verbose : bool, default=False
        Whether to print progress bar while training
    """

    def __init__(
        self,
        n_iter=100,
        n_factors=10,
        learning_rate=0.01,
        reg_w=0.01,
        reg_v=0.01,
        random_state=None,
        verbose=False,
    ):
        self.n_iter = n_iter
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.reg_w = reg_w
        self.reg_v = reg_v
        self.random_state = random_state
        self.verbose = verbose

    def fit(self, X, y):
        """
        Fit the model according to the given training data

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features
        y : array-like of shape (n_samples,)
            Target vector relative to X

        Returns
        -------
        self : object
            Returns self
        """
        self._setup_input(X, y)
        self._init_weights()

        for _ in tqdm(range(self.n_iter), disable=not self.verbose):
            self._update_weights()

        return self

    def predict(self, X):
        """
        Predict using the linear model

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        C : array-like of shape (n_samples,)
            Returns predicted values
        """
        return self._predict(X)

    def _setup_input(self, X, y):
        self.X = np.array(X)
        self.y = np.array(y)
        self.n_samples, self.n_features = self.X.shape

    def _init_weights(self):
        # Bias
        self.w0 = 0
        # Feature weights
        self.w = np.zeros(self.n_features)
        # Factorized weights
        self.V = np.random.normal(
            scale=1 / self.n_factors, size=(self.n_features, self.n_factors)
        )  # (n_features, n_factors)

    def _update_weights(self):
        y_pred = self._predict(self.X)
        loss = self.loss_grad(self.y, y_pred)
        w_grad = np.dot(loss, self.X) / self.n_samples

        # Update bias
        self.w0 -= self.learning_rate * loss.mean() + 2 * self.reg_w * self.w0

        # Update feature weights
        self.w -= self.learning_rate * w_grad + 2 * self.reg_w * self.w

        # Update factorized weights
        for i in range(self.n_features):
            for f in range(self.n_factors):
                v_grad = (
                    loss * (self.X[:, i] * (self.X.dot(self.V)[:, f] - self.X[:, i] * self.V[i, f]))
                ).mean()
                self.V[i, f] -= self.learning_rate * v_grad + 2 * self.reg_v * self.V[i, f]

    @property
    def loss_grad(self):
        raise NotImplementedError
    
    @loss_grad.setter
    def loss_grad(self, loss_grad):
        self._loss_grad = loss_grad
    
    @loss_grad.getter
    def loss_grad(self):
        return self._loss_grad

    def _predict(self, X):
        return self.w0 + X.dot(self.w) + self._predict_factorized(X)

    def _predict_factorized(self, X):
        return (
            np.sum(np.power(X.dot(self.V), 2) - X.dot(np.power(self.V, 2)), axis=1) / 2
        )

class FMRegressor(FMBase): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss = "mse"
        self.loss_grad = elementwise_grad(mean_squared_error)

class FMClassifier(FMBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._loss = "logloss"
        self.loss_grad = elementwise_grad(binary_crossentropy)

    def predict_proba(self, X):
        """
        Probability estimates

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Samples

        Returns
        -------
        C : array-like of shape (n_samples,)
            Returns predicted values
        """
        return self._predict_proba(X)
    
    def _predict_proba(self, X):
        return self._sigmoid(self._predict(X))
    
    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
    def predict(self, X):
        predictions = self._predict(X)
        return np.sign(predictions)
