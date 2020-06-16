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
        raise ValueError(
            "Cannot select {} rows from matrix X; it has only {}".format(k, n)
        )
    ind = subsample_ind(n, k, seed)
    return X[ind, :]


def tr_te_indices(n, tr_proportion, seed=9282):
    """Get two logical vectors for indexing train/test points.

    Return (tr_ind, te_ind)
    """
    if not 0.0 <= tr_proportion <= 1.0:
        raise ValueError(
            "tr_proportion must be in [0, 1]; found {}".format(tr_proportion)
        )

    Itr = np.full(n, False)
    tr_ind = subsample_ind(n, int(tr_proportion * n), seed)
    Itr[tr_ind] = True
    return Itr, ~Itr


def median_distance(X):
    """
    Compute the median of pairwise distances of points in the matrix.

    Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array

    Return
    ------
    The median distance. If it is nonpositive, return the mean. This can happen
    e.g. when the data are 0s and 1s and there are more 0s than 1s.
    """
    D = dist_matrix(X, X)
    Itri = np.tril_indices(D.shape[0], -1)
    Tri = D[Itri]
    med = np.median(Tri)
    return med if med > 0 else np.mean(Tri)


def sampled_median_distance(X, subsample, seed=9827):
    """
    Compute the subsampled median of pairwise distances of points in the matrix.

    Useful as a heuristic for setting Gaussian kernel's width.

    Parameters
    ----------
    X : n x d numpy array
    subsample: number of points to sample from X to determine median

    Return
    ------
    The subsampled median distance.
    """
    if subsample <= 0:
        raise ValueError(
            "subsample must be positive; found {}".format(subsample)
        )

    n = X.shape[0]
    if subsample > n:
        subsample = n

    Xi = subsample_rows(X, subsample, seed)
    return median_distance(Xi)


def is_real_num(x):
    """Return true iff x is a real number."""
    return np.isscalar(x) and np.isfinite(x) and np.isrealobj(x)


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
    nx, dx = X.shape
    ny, dy = Y.shape
    if nx != ny:
        raise ValueError("X has {} rows while Y has {} rows".format(nx, ny))

    mx = np.mean(X, 0)
    my = np.mean(Y, 0)

    Cxy = X.T.dot(Y)  # dx x dy
    np.divide(Cxy, nx, out=Cxy)
    np.subtract(Cxy, np.outer(mx, my), out=Cxy)

    # Cxx, Cyy have to be invertible
    Cxx = np.cov(X.T)
    Cyy = np.cov(Y.T)

    if dx == 1:
        CxxICxy = Cxy / Cxx
    else:
        regmat = np.identity(dx)
        np.multiply(regmat, reg, out=regmat)
        np.add(Cxx, regmat, out=Cxx)
        CxxICxy = np.linalg.solve(Cxx, Cxy)

    if dy == 1:
        CyyICyx = Cxy.T / Cyy
    else:
        regmat = np.identity(dy)
        np.multiply(regmat, reg, out=regmat)
        np.add(Cyy, regmat, out=Cyy)
        CyyICyx = np.linalg.solve(Cyy, Cxy.T)

    # Problems for a and b:
    avals, aV = np.linalg.eig(CxxICxy.dot(CyyICyx))
    bvals, bV = np.linalg.eig(CyyICyx.dot(CxxICxy))

    dim = min(dx, dy)

    # Sort in descending order and select first `dim` entries
    Ia = np.argsort(-avals)[:dim]
    avals = avals[Ia]
    aV = aV[:, Ia]

    Ib = np.argsort(-bvals)[:dim]
    bvals = bvals[Ib]
    bV = bV[:, Ib]

    return np.real(avals), np.real(aV), np.real(bV)


def sym_to_power(X, power, fix=0):
    """
    Raise symmetric matrix to given power through eigenvalue decomposition.
    """
    # Since X is symmetric, use `eigh` decomposition.
    evals, evecs = np.linalg.eigh(X)

    # If X is full rank, all eigenvalues are positive, but we can optionally
    # ensure the eigenvalues are non-zero. This is usefull in case `power` < 0.
    np.maximum(0, evals, out=evals)
    if fix != 0:
        np.add(evals, fix, out=evals)

    # Since the matrix is symemtric, the eigenvectors should be real.
    evecs = np.real(evecs)

    np.power(evals, power, out=evals)

    Y = evecs * evals
    return np.dot(Y, evecs.T, out=Y)


def fit_gaussian_draw(X, J, seed=28, reg=1e-7, eig_pow=1.0):
    """
    Fit a multivariate normal to X (n x d) and draw J points from the fit.

    - reg: regularizer to use with the covariance matrix
    - eig_pow: raise eigenvalues of the covariance matrix to this power to
        construct a new covariance matrix before drawing samples. Useful to
        shrink the spread of the variance.
    """
    with NumpySeedContext(seed=seed):
        d = X.shape[1]
        cov_x = np.cov(X, rowvar=True)  # construct the d x d covariance matrix
        if d == 1:
            # TODO: Write a unittest for this case!
            cov_x = np.array([[cov_x]])

        shrunk_cov = sym_to_power(cov_x, eig_pow)

        # Add regularizer to shrunken covariance matrix.
        regmat = np.identity(d)
        np.multiply(regmat, reg, out=regmat)
        np.add(shrunk_cov, regmat, out=shrunk_cov)

        return np.random.multivariate_normal(np.mean(X, 0), shrunk_cov, J)


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
