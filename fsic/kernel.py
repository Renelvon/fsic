"""Various kernels"""

import abc

import numpy as np
from scipy import signal


class Kernel(metaclass=abc.ABCMeta):
    """Abstract class for kernels"""

    @abc.abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """


class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form (x.dot(y))**d"""

    def __init__(self, degree):
        if degree <= 0:
            raise ValueError("degree must be positive; found {}".format(degree))

        self.degree = degree

    def eval(self, X1, X2):
        return X1.dot(X2.T) ** self.degree

    def __repr__(self):
        return "KHoPoly(degree={})".format(self.degree)


class KLinear(Kernel):
    def eval(self, X1, X2):
        return X1.dot(X2.T)

    def __repr__(self):
        return "KLinear()"


class KGauss(Kernel):
    def __init__(self, sigma2):
        if sigma2 <= 0:
            raise ValueError("sigma2 must be positive; found {}".format(sigma2))

        self.sigma2 = sigma2

    def eval(self, X1, X2):
        """
        Evaluate the Gaussian kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x d numpy array
        X2 : n2 x d numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        _, d1 = X1.shape
        _, d2 = X2.shape
        if d1 != d2:
            raise ValueError(
                "The X1 dimensions (_, {}) do not match the X2 dimensions (_, {})".format(d1, d2)
            )

        D2 = X1.dot(X2.T)
        np.multiply(D2, -2, out=D2)
        np.add(D2, np.sum(X1 ** 2, 1, keepdims=True), out=D2)
        np.add(D2, np.sum(X2 ** 2, 1), out=D2)
        np.divide(D2, -self.sigma2, out=D2)

        return np.exp(D2)

    def __repr__(self):
        return "KGauss(sigma2={})".format(self.sigma2)


class KTriangle(Kernel):
    """
    A triangular kernel defined on 1D. k(x, y) = B_1((x-y)/width) where B_1 is the
    B-spline function of order 1 (i.e., triangular function).
    """

    def __init__(self, width):
        if width <= 0:
            raise ValueError("width must be positive; found {}".format(width))

        self.width = width

    def eval(self, X1, X2):
        """
        Evaluate the triangular kernel on the two 2d numpy arrays.

        Parameters
        ----------
        X1 : n1 x 1 numpy array
        X2 : n2 x 1 numpy array

        Return
        ------
        K : a n1 x n2 Gram matrix.
        """
        _, d1 = X1.shape
        if d1 != 1:
            raise ValueError("The X1 dimension (_, {}) must be 1".format(d1))

        _, d2 = X2.shape
        if d2 != 1:
            raise ValueError("The X2 dimension (_, {}) must be 1".format(d2))

        diff = (X1 - X2.T)
        np.divide(diff, self.width, out=diff)
        return signal.bspline(diff, 1)

    def __repr__(self):
        return "KTriangle(width={})".format(self.width)
