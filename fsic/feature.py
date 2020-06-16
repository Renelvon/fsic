"""Module containing classes for extracting/constructing features from data"""

import abc

import numpy as np
from scipy import stats


class FeatureMap(metaclass=abc.ABCMeta):
    """Abstract class for a feature map function"""

    @abc.abstractmethod
    def gen_features(self, X):
        """Generate D features for each point in X.
        - X: nxd data matrix

        Return a n x D numpy array.
        """

    @abc.abstractmethod
    def num_features(self, X=None):
        """
        Return the number of features that this map will generate for X.
        X is optional.
        """


class MarginalCDFMap(FeatureMap):
    """
    A FeatureMap that returns a new set of variates generated by applying
    the empirical CDF of each variate to its corresponding variate.
    Also called, a copula transform or a probability integral transform.
    """

    def gen_features(self, X):
        """
        Cost O(d * n * log(n)) where X is n x d.
        """
        n, d = X.shape
        Z = np.empty_like(X)
        for j in range(d):
            Z[:, j] = stats.rankdata(X[:, j])
        Z /= n
        return Z

    def num_features(self, X=None):
        if X is None:
            raise ValueError("Cannot compute features without an X")
        return X.shape[1]


class RFFKGauss(FeatureMap):
    """
    A FeatureMap to construct random Fourier features for a Gaussian kernel.
    """

    def __init__(self, sigma2, n_features, seed=20):
        """
        n_features: number of random Fourier features. The total number of
            dimensions will be n_features*2.
        """
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive; found {}".format(sigma2))

        if n_features <= 0:
            raise ValueError("n_features must be positive; found {}".format(n_features))

        self.sigma2 = sigma2
        self.n_features = n_features
        self.seed = seed

    def gen_features(self, X):
        rstate = np.random.get_state()
        np.random.seed(self.seed)
        _, d = X.shape

        D = self.n_features
        W = np.random.randn(D, d)
        # n x D
        XWT = X.dot(W.T) / np.sqrt(self.sigma2)
        Z1 = np.cos(XWT)
        Z2 = np.sin(XWT)
        Z = np.hstack((Z1, Z2)) * np.sqrt(1.0 / self.n_features)

        np.random.set_state(rstate)
        return Z

    def num_features(self, X=None):
        return 2 * self.n_features


class NystromFeatureMap(FeatureMap):
    """
    A FeatureMap to construct features Z (n x D) such that Z.dot(Z.T) gives
    a good approximation to the kernel matrix K constructed by using the
    specified kernel k.

    Procedure
    - A subset of D inducing points is given.
    - Form an n x D kernel matrix K between the input points and the inducing
      points.
    - Form a D x D kernel matrix M of the inducing points.
    - Features = K.dot(M**-0.5) (matrix power)
    """

    def __init__(self, k, inducing_points):
        """
        k: a Kernel
        inducing_points: a D x d matrix. D = number of points. d = dimensions.
            The number of features is D.
        """
        self.k = k
        self.inducing_points = inducing_points
        # a cache to make it faster
        M = k.eval(inducing_points, inducing_points)
        # eigen decompose. Want to raise to the power of -0.5
        evals, V = np.linalg.eig(M)
        # Assume M is full rank
        pow_evals = 1.0 / np.sqrt(evals + 1e-6)
        self._invert_half = V.dot(np.diag(pow_evals)).dot(V.T)

    def gen_features(self, X):
        _, d = X.shape
        if d != self.inducing_points.shape[1]:
            raise ValueError(
                "dimension of the input does not match that of the inducing points"
            )
        K = self.k.eval(X, self.inducing_points)
        Z = K.dot(self._invert_half)
        return Z

    def num_features(self, X=None):
        return self.inducing_points.shape[1]
