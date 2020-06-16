"""Convenient methods for general machine learning"""

import numpy as np


class NumpySeedContext:
    """
    A context manager to reset the random seed by numpy.random.seed(..).
    Set the seed back at the end of the block.
    """

    def __init__(self, seed):
        self.seed = seed
        self.cur_state = None

    def __enter__(self):
        rstate = np.random.get_state()
        self.cur_state = rstate
        np.random.seed(self.seed)
        return self

    def __exit__(self, *args):
        np.random.set_state(self.cur_state)


def dist_matrix2(X, Y):
    """
    Construct a pairwise squared Euclidean distance matrix of
    size X.shape[0] x Y.shape[0]
    """
    D = X.dot(Y.T)
    np.multiply(D, -2, out=D)
    np.add(D, np.sum(X ** 2, 1, keepdims=True), out=D)
    return np.add(D, np.sum(Y ** 2, 1), out=D)


def dist_matrix(X, Y):
    """
    Construct a pairwise Euclidean distance matrix of
    size X.shape[0] x Y.shape[0]
    """
    D = dist_matrix2(X, Y)

    # Clamp negative numbers to 0, to avoid errors from taking sqrt.
    np.maximum(D, 0, out=D)

    return np.sqrt(D, out=D)


def meddistance(X, subsample=None, mean_on_fail=True):
    """
    Compute the median of pairwise distances (not distance squared) of points
    in the matrix.  Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    mean_on_fail: True/False. If True, use the mean when the median distance is 0.
        This can happen especially, when the data are discrete e.g., 0/1, and
        there are more slightly more 0 than 1. In this case, the m

    Return
    ------
    median distance
    """
    if subsample is None:
        D = dist_matrix(X, X)
        Itri = np.tril_indices(D.shape[0], -1)
        Tri = D[Itri]
        med = np.median(Tri)
        if med <= 0:
            # use the mean
            return np.mean(Tri)
        return med

    assert subsample > 0
    rand_state = np.random.get_state()
    np.random.seed(9827)
    n = X.shape[0]
    ind = np.random.choice(n, min(subsample, n), replace=False)
    np.random.set_state(rand_state)
    # recursion just one
    return meddistance(X[ind, :], None, mean_on_fail)


def is_real_num(x):
    """Return true iff x is a real number."""
    return np.isscalar(x) and np.isfinite(x) and np.isrealobj(x)


def tr_te_indices(n, tr_proportion, seed=9282):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    Itr = np.zeros(n, dtype=bool)
    tr_ind = np.random.choice(n, int(tr_proportion * n), replace=False)
    Itr[tr_ind] = True
    Ite = np.logical_not(Itr)

    np.random.set_state(rand_state)
    return (Itr, Ite)


def subsample_ind(n, k, seed=32):
    """
    Return a list of indices to choose k out of n without replacement
    """
    rand_state = np.random.get_state()
    np.random.seed(seed)

    ind = np.random.choice(n, k, replace=False)
    np.random.set_state(rand_state)
    return ind


def subsample_rows(X, k, seed=29):
    """
    Subsample k rows from the matrix X.
    """
    n = X.shape[0]
    if k > n:
        raise ValueError("k exceeds the number of rows.")
    ind = subsample_ind(n, k, seed=seed)
    return X[ind, :]


def cca(X, Y, reg=1e-5):
    """
    CCA formulation solving two eigenvalue problems.

    - X: n x dx data matrix
    - Y: n x dy data matrix

    Return (vals, Vx, Vy) where
        vals is a numpy array of decreasing eigenvalues,
        Vx is a square matrix whose columns are eigenvectors for X corresponding to vals.
        Vy is a square matrix whose columns are eigenvectors for Y corresponding to vals.
    """
    dx = X.shape[1]
    dy = Y.shape[1]
    assert X.shape[0] == Y.shape[0]
    n = X.shape[0]
    mx = np.mean(X, 0)
    my = np.mean(Y, 0)
    # dx x dy
    Cxy = X.T.dot(Y) / n - np.outer(mx, my)
    Cxx = np.cov(X.T)
    Cyy = np.cov(Y.T)
    # Cxx, Cyy have to be invertible

    if dx == 1:
        CxxICxy = Cxy / Cxx
    else:
        CxxICxy = np.linalg.solve(Cxx + reg * np.eye(dx), Cxy)

    if dy == 1:
        CyyICyx = Cxy.T / Cyy
    else:
        CyyICyx = np.linalg.solve(Cyy + reg * np.eye(dy), Cxy.T)

    # problem for a
    avals, aV = np.linalg.eig(CxxICxy.dot(CyyICyx))
    # problem for b
    bvals, bV = np.linalg.eig(CyyICyx.dot(CxxICxy))

    dim = min(dx, dy)
    # sort descendingly
    Ia = np.argsort(-avals)
    avals = avals[Ia[:dim]]
    aV = aV[:, Ia[:dim]]

    Ib = np.argsort(-bvals)
    bvals = bvals[Ib[:dim]]
    bV = bV[:, Ib[:dim]]
    np.testing.assert_array_almost_equal(avals, bvals)
    return np.real(avals), np.real(aV), np.real(bV)


def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    """
    Fit a multivariate normal to the data X (n x d) and draw J points
    from the fit.
    - reg: regularizer to use with the covariance matrix
    - eig_pow: raise eigenvalues of the covariance matrix to this power to construct
        a new covariance matrix before drawing samples. Useful to shrink the spread
        of the variance.
    """
    with NumpySeedContext(seed=seed):
        d = X.shape[1]
        mean_x = np.mean(X, 0)
        cov_x = np.cov(X.T)
        if d == 1:
            cov_x = np.array([[cov_x]])
        [evals, evecs] = np.linalg.eig(cov_x)
        evals = np.maximum(0, np.real(evals))
        assert np.all(np.isfinite(evals))
        evecs = np.real(evecs)
        shrunk_cov = evecs.dot(np.diag(evals ** eig_pow)).dot(
            evecs.T
        ) + reg * np.eye(d)
        V = np.random.multivariate_normal(mean_x, shrunk_cov, J)
    return V


def bound_by_data(Z, Data):
    """
    Determine lower and upper bound for each dimension from the Data, and project
    Z so that all points in Z live in the bounds.

    Z: m x d
    Data: n x d

    Return a projected Z of size m x d.
    """
    n, _ = Z.shape
    Low = np.min(Data, 0)
    Up = np.max(Data, 0)
    LowMat = np.repeat(Low[np.newaxis, :], n, axis=0)
    UpMat = np.repeat(Up[np.newaxis, :], n, axis=0)

    Z = np.maximum(LowMat, Z)
    Z = np.minimum(UpMat, Z)
    return Z


def one_of_K_code(arr):
    """
    Make a one-of-K coding out of the numpy array.
    For example, if arr = ([0, 1, 0, 2]), then return a 2d array of the form
     [[1, 0, 0],
      [0, 1, 0],
      [1, 0, 0],
      [0, 0, 2]]
    """
    U = np.unique(arr)
    n = len(arr)
    nu = len(U)
    X = np.zeros((n, nu))
    for i, u in enumerate(U):
        Ii = np.where(np.abs(arr - u) < 1e-8)
        X[Ii[0], i] = 1
    return X


def standardize(X):
    mx = np.mean(X, 0)
    stdx = np.std(X, axis=0)
    # Assume standard deviations are not 0
    Zx = (X - mx) / stdx
    assert np.all(np.isfinite(Zx))
    return Zx
