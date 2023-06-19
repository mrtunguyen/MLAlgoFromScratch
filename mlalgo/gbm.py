import numpy as np
from scipy.special import expit


class Tree:
    def __init__(self, regression=True, criterion=None, n_classes=None):
        self.regression = regression
        self.criterion = criterion
        self.n_classes = n_classes
        self.loss = None
        self.imputiry = None
        self.threshold = None

        self.left_child = None
        self.right_child = None

    def fit(
        self,
        X,
        target,
        max_depth=2,
        max_features: int = 100,
        min_samples_split: int = 2,
        min_samples_leaf=1,
        loss=None,
    ):
        if loss is not None:
            self.loss = loss

        if not self.regression:
            self.n_classes = len(np.unique(target))

        self._fit(
            X, target, max_depth, max_features, min_samples_split, min_samples_leaf
        )

    def _fit(
        self,
        X,
        target,
        max_depth: int,
        max_features: int,
        min_samples_split: int,
        min_samples_leaf: int,
        minimum_gain: float = 0.01,
    ):
        try:
            assert (
                X.shape[0] > min_samples_split
            ), "X.shape[0] must be greater than min_samples_split"
            assert max_depth > 0, "max_depth must be greater than 0"

            if max_features is None:
                max_features = X.shape[1]

            column, threshold, gain = self._find_best_split(X, target, max_features)

            assert gain is not None

            if self.regression:
                assert gain >= 0.0
            else:
                assert gain > minimum_gain

            self.column = column
            self.threshold = threshold
            self.imputiry = gain

            # Split the data
            left_X, right_X, left_target, right_target = self._split_data(
                X, target, column, threshold
            )

            # Build the tree
            self.left = Tree(self.regression, self.criterion)
            self.left._fit(
                left_X,
                left_target,
                max_depth - 1,
                max_features,
                min_samples_split,
                min_samples_leaf,
            )

            self.right = Tree(self.regression, self.criterion)
            self.right._fit(
                right_X,
                right_target,
                max_depth - 1,
                max_features,
                min_samples_split,
                min_samples_leaf,
            )

        except AssertionError:
            self._build_leaf(target)

    def _build_leaf(self, target):
        """find the best leaf value"""
        if self.loss is not None:  # gradient boosting
            self.outcome = self.loss.approximate(target["actual"], target["predicted"])

        else:
            if self.regression:
                self.outcome = np.mean(target["y"])
            else:
                self.outcome = (
                    np.bincount(target["y"], minlength=self.n_classes)
                    / target["y"].shape[0]
                )

    def _find_best_split(self, X, target, max_features):
        subset_features = np.random.choice(
            X.shape[1], max_features, replace=False
        ).tolist()
        max_gain, max_column, max_threshold = -np.inf, None, None

        for column in subset_features:
            split_values = self._find_splits(X[:, column])
            for value in split_values:
                if self.loss is not None:
                    # gradient boosting
                    left, right = self._split_data(X, target, column, value)
                    gain = self.criterion(target, left, right, self.loss)
                else:
                    # random forest
                    splits = split(X[:, column], target["y"], value)
                    gain = self.criterion(target, splits)

                if (max_gain is None) or (gain > max_gain):
                    max_gain = gain
                    max_column = column
                    max_threshold = value
        return max_column, max_threshold, max_gain

    def _find_splits(self, X):
        """Find all possible split values given the data."""
        X = np.unique(X)
        return (X[1:] + X[:-1]) / 2.0

    @staticmethod
    def _split_data(X, target, column, threshold, return_X=False):
        """Split the data based on the threshold."""
        left_mask = X[:, column] < threshold
        right_mask = X[:, column] >= threshold

        left, right = {}, {}
        for key in target.keys():
            left[key] = target[key][left_mask]
            right[key] = target[key][right_mask]

        if return_X:
            left_X, right_X = X[left_mask], X[right_mask]
            return left_X, right_X, left, right
        else:
            return left, right


class GradientBoosting:
    """Gradient Boosting algorithm implementation."""

    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=2,
        max_features=10,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=1.0,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.trees = []
        self.loss = None

    def fit(self, X, y):
        """Fit the model to the data X and target y."""
        self._setup_input(X, y)
        self.y_mean = np.mean(y)
        self._fit()

    def _setup_input(self, X, y):
        """Setup the input data."""
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape

    def _fit(self):
        """Train the model."""
        # Initialize model with zeros
        y_pred = np.zeros(self.n_samples, np.float32)

        for n in range(self.n_estimators):
            residuals = self.loss.grad(self.y, y_pred)
            tree = Tree(regression=True, criterion=mse_criterion)
            targets = {"actual": self.y, "predicted": y_pred, "y": residuals}
            tree.fit(
                self.X,
                targets,
                max_depth=self.max_depth,
                max_features=self.max_features,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                loss=self.loss,
            )
            y_pred += self.learning_rate * tree.predict(self.X)
            self.trees.append(tree)

    def _predict(self, X):
        """Predict the output given the input."""
        y_pred = np.zeros(X.shape[0], np.float32)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred

    def predict(self, X):
        """Predict the output given the input."""
        return self.loss.transform(self._predict(X))


class Loss:
    def __init__(self, regularization=0.5):
        self.regularization = regularization

    def grad(self, y, y_pred):
        raise NotImplementedError

    def hess(self, y, y_pred):
        raise NotImplementedError

    def transform(self, y_pred):
        raise y_pred
    
    def approximate(self, y, y_pred):
        return self.grad(y, y_pred).sum() / (self.hess(y, y_pred).sum() + self.regularization)
    
    def gain(self, y, y_pred):
        return 0.5 * (self.grad(y, y_pred).sum() ** 2) / (self.hess(y, y_pred).sum() + self.regularization)


class LeastSquaresLoss(Loss):

    def grad(self, y, y_pred):
        return y - y_pred

    def hess(self, y, y_pred):
        return np.ones(y.shape[0])


class LogisticLoss(Loss):

    def grad(self, y, y_pred):
        return y * expit(-y * y_pred)

    def hess(self, y, y_pred):
        expits = expit(y_pred)
        return expits * (1 - expits)

    def transform(self, y_pred):
        return expit(y_pred)


class GradientBoostingRegressor(GradientBoosting):
    def fit(self, X, y):
        self.loss = LeastSquaresLoss()
        super().fit(X, y)


class GradientBoostingClassifier(GradientBoosting):
    def fit(self, X, y):
        # Convert y to {-1, 1}
        y = y * 2 - 1
        self.loss = LogisticLoss()
        super().fit(X, y)
