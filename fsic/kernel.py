"""Module containing kernel related classes"""

import abc

import numpy as np
import scipy.signal as sig


class Kernel(metaclass=abc.ABCMeta):
    """Abstract class for kernels"""

    @abc.abstractmethod
    def eval(self, X1, X2):
        """Evalute the kernel on data X1 and X2 """


class KHoPoly(Kernel):
    """Homogeneous polynomial kernel of the form
    (x.dot(y))**d
    """

    def __init__(self, degree):
        assert degree > 0
        self.degree = degree

    def eval(self, X1, X2):
        return X1.dot(X2.T) ** self.degree

    def __str__(self):
        return "KHoPoly(d=%d)" % self.degree


class KLinear(Kernel):
    def eval(self, X1, X2):
        return X1.dot(X2.T)

    def __str__(self):
        return "KLinear()"


class KGauss(Kernel):
    def __init__(self, sigma2):
        assert sigma2 > 0, "sigma2 must be > 0. Was %s" % str(sigma2)
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
        assert d1 == d2, "Dimensions of the two inputs must be the same"
        D2 = (
            np.sum(X1 ** 2, 1)[:, np.newaxis]
            - 2 * X1.dot(X2.T)
            + np.sum(X2 ** 2, 1)
        )
        K = np.exp(-D2 / self.sigma2)
        return K

    def __str__(self):
        return "KGauss(%.3f)" % self.sigma2


class KTriangle(Kernel):
    """
    A triangular kernel defined on 1D. k(x, y) = B_1((x-y)/width) where B_1 is the
    B-spline function of order 1 (i.e., triangular function).
    """

    def __init__(self, width):
        assert width > 0, "width must be > 0"
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
        _, d2 = X2.shape
        assert d1 == 1, "d1 must be 1"
        assert d2 == 1, "d2 must be 1"
        diff = (X1 - X2.T) / self.width
        K = sig.bspline(diff, 1)
        return K

    def __str__(self):
        return "KTriangle(w=%.3f)" % self.width
