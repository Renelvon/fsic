import abc
import math

import numpy as np
from scipy import stats

from fsic import util


class PairedData:
    """
    Paired data for independence testing

    Attributes:
    -----------
    X: numpy array
    Y: numpy array

    X and Y are pairs of equal sample sizes (rows) but may differ in dimensions (columns).
    """

    def __init__(self, X, Y, label=""):
        nx = X.shape[0]
        ny = Y.shape[0]

        if nx != ny:
            raise ValueError(
                "The matrices must contain the same amount of samples; matrix X has {} rows while matrix Y has {}".format(
                    nx, ny
                )
            )

        self._X = X
        self._Y = Y

        # Short description to be used as a plot label
        self.label = label

        if not np.all(np.isfinite(X)):
            raise ValueError("Some element of matrix X is infinite or NaN")

        if not np.all(np.isfinite(Y)):
            raise ValueError("Some element of matrix Y is infinite or NaN")

    @property
    def X(self):
        return self._X

    @property
    def Y(self):
        return self._Y

    @staticmethod
    def mean_and_std_of(Z):
        return np.mean(Z, 0), np.std(Z, 0)

    @staticmethod
    def array_to_str(Z, precision=4):
        return np.array_str(Z, precision=precision)

    def summary(self, prec=4):
        mx, stdx = self.mean_and_std_of(self._X)
        my, stdy = self.mean_and_std_of(self._Y)

        return "E[x] = {} \n Std[x] = {} \n E[y] = {} \n Std[y] = {} \n".format(
            self.array_to_str(mx),
            self.array_to_str(stdx),
            self.array_to_str(my),
            self.array_to_str(stdy),
        )

    @property
    def dx(self):
        """Return the dimension of X."""
        return self._X.shape[1]

    @property
    def dy(self):
        """Return the dimension of Y."""
        return self._Y.shape[1]

    @property
    def sample_size(self):
        return self._X.shape[0]

    @property
    def xy(self):
        """Return X and Y as a tuple"""
        return self._X, self._Y

    def split_tr_te(self, tr_proportion=0.5, seed=820):
        """
        Split the dataset into training and test sets.

        Return (PairedData for tr, PairedData for te).
        """
        X, Y = self.xy
        Itr, Ite = util.tr_te_indices(X.shape[0], tr_proportion, seed)

        label = self.label
        tr_data = PairedData(X[Itr, :], Y[Itr, :], "tr_{}".format(label))
        te_data = PairedData(X[Ite, :], Y[Ite, :], "te_{}".format(label))

        return tr_data, te_data

    def subsample(self, n, seed=87):
        """Sample this PairedData without replacement; return new PairedData."""
        nx = self.sample_size
        if n > nx:
            raise ValueError(
                "Sample cannot be larger than size of X and Y ({}); found {}".format(
                    nx, n
                )
            )
        ind = util.subsample_ind(nx, n, seed)
        return PairedData(self._X[ind, :], self._Y[ind, :], self.label)

    def __add__(self, pdata):
        """Merge the current PairedData with another one."""
        nX = np.vstack((self._X, pdata._X))
        nY = np.vstack((self._Y, pdata._Y))
        nlabel = "{}_{}".format(self.label, pdata.label)
        return PairedData(nX, nY, nlabel)


class PairedSource(metaclass=abc.ABCMeta):
    """
    A data source that can be resampled.

    Subclasses may prefix class names with PS. If possible, prefix with PSInd
    to indicate that the PairedSource contains two independent samples; prefix
    with PSDep otherwise. Use PS if the PairedSource can be either one
    depending on the provided parametres.
    """

    @abc.abstractmethod
    def sample(self, n, seed):
        """
        Return a PairedData.

        The result should be deterministic given the input (n, seed).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def dx(self):
        """Return the dimension of X"""
        raise NotImplementedError

    @abc.abstractmethod
    def dy(self):
        """Return the dimension of Y"""
        raise NotImplementedError


class FinitePairedSource(PairedSource, metaclass=abc.ABCMeta):
    """
    A PairedSource which generates output through sampling a known initial set.
    """

    def __init__(self, pdata):
        """
        pdata: a PairedData object
        """
        self.pdata = pdata

    def dx(self):
        """Return the dimension of X"""
        return self.pdata.dx

    def dy(self):
        """Return the dimension of Y"""
        return self.pdata.dy


class PSStraResample(FinitePairedSource):
    """
    A source which does stratified subsampling without replacement.

    The implementation is only approximately correctly.
    """

    def __init__(self, pdata, pivot):
        """
        pdata: a PairedData object
        pivot: one-dimensional numpy array indicating the class of each point;
            it must have as many entries as pdata.sample_size
        """
        super().__init__(pdata)

        lp = len(pivot)
        nx = pdata.sample_size

        if lp != nx:
            raise ValueError(
                "The pivot has {} entries; it must have as many as samples in the PairedData: {}".format(
                    lp, nx
                )
            )

        self.pivot = pivot

        self._uniques, self._counts = np.unique(pivot, return_counts=True)

    def sample(self, n, seed=900):
        X, Y = self.pdata.xy
        nx = X.shape[0]

        # Permute X, Y, pivot.
        I = util.subsample_ind(nx, nx, seed=seed + 3)
        X = X[I, :]
        Y = Y[I, :]
        pivot = self.pivot[I]

        # Choose at least 1 instance from each class.
        class_counts = tuple(
            int(math.ceil(count / nx * n)) for count in self._counts
        )

        idxs = np.concatenate(
            tuple(
                np.nonzero(pivot == value)[0][:count]
                for value, count in zip(self._uniques, class_counts)
            )
        )

        ix = idxs.shape[0]
        if ix > n:
            idxs = idxs[util.subsample_ind(ix, n, seed + 5)]

        return PairedData(X[idxs, :], Y[idxs, :], self.pdata.label + "_stra")


class PSNullResample(FinitePairedSource):
    """
    A source which subsamples without replacement, and then randomly permutes
    the order of one sample, to break pairs.

    This is meant to simulate the case where [H0: X, Y are independent] holds.
    """

    def sample(self, n, seed=981):
        pdata = self.pdata.subsample(n, seed=seed)
        nX, Y = pdata.xy
        nY = np.roll(Y, 1, 0)
        return PairedData(nX, nY, self.pdata.label + "_shuf")


class PSGaussNoiseDims(PairedSource):
    """
    A PairedSource that adds noise dimensions to X, Y drawn from the specified
    PairedSource. The noise follows the standard normal distribution.

    Decorator pattern.
    """

    def __init__(self, ps, ndx, ndy):
        """
        ps: a PairedSource
        ndx: number of noise dimensions for X
        ndy: number of noise dimensions for Y
        """
        if ndx < 0:
            raise ValueError(
                "The noise dimensions for X must be non-negative; found {}".format(
                    ndx
                )
            )

        if ndy < 0:
            raise ValueError(
                "The noise dimensions for Y must be non-negative; found {}".format(
                    ndy
                )
            )

        self.ps = ps
        self.ndx = ndx
        self.ndy = ndy

    def sample(self, n, seed=44):
        with util.NumpySeedContext(seed=seed + 100):
            NX = np.random.randn(n, self.ndx)
            NY = np.random.randn(n, self.ndy)

        pdata = self.ps.sample(n, seed=seed)
        X, Y = pdata.xy
        Zx = np.hstack((X, NX))
        Zy = np.hstack((Y, NY))
        new_label = "{}_ndx{}_ndy{}".format(pdata.label, self.ndx, self.ndy)
        return PairedData(Zx, Zy, label=new_label)

    def dx(self):
        return self.ps.dx() + self.ndx

    def dy(self):
        return self.ps.dy() + self.ndy


class PSFunc(PairedSource):
    """
    A PairedSource that generates data (X, Y) such that Y = f(X) for a
    specified function f (possibly stochastic), and px where X ~ px.
    """

    def __init__(self, f, px):
        """
        f: a function accepting an (n x dx) array X and returning an (n x dy) array Y
        px: prior used to generate X; accepts integer n and returns (n x dx) array X
        """
        self.f = f
        self.px = px

        # Discover output dimensions by invoking px and f; discard result.
        X = px(2)
        self._dx = X.shape[1]
        self._dy = f(X).shape[1]

    def sample(self, n, seed):
        with util.NumpySeedContext(seed=seed):
            X = self.px(n)
            Y = self.f(X)
        return PairedData(X, Y, label="psfunc")

    def dx(self):
        return self._dx

    def dy(self):
        return self._dy


class PSIndUnif(PairedSource):
    """
    Multivariate (or univariate) uniform distributions for both X, Y
    on the specified boundaries
    """

    def __init__(self, xlb, xub, ylb, yub):
        """
        xlb: a numpy array of lower bounds of x
        xub: a numpy array of upper bounds of x
        ylb: a numpy array of lower bounds of y
        yub: a numpy array of upper bounds of y
        """

        def convertif(a):
            return np.array(a) if isinstance(a, list) else a

        xlb, xub, ylb, yub = map(convertif, [xlb, xub, ylb, yub])

        if xlb.shape[0] != xub.shape[0]:
            raise ValueError(
                "Lower bounds for X are {} but upper bounds are {}; they must match".format(
                    xlb.shape, xub.shape
                )
            )

        if not np.all(xlb < xub):
            raise ValueError(
                "Lower bounds for some dimensions of X are not smaller than upper bounds"
            )

        self.xlb = xlb
        self.xub = xub
        self._pscale = xub - xlb

        if ylb.shape[0] != yub.shape[0]:
            raise ValueError(
                "Lower bounds for Y are {} but upper bounds are {}; they must match".format(
                    ylb.shape, yub.shape
                )
            )

        if not np.all(ylb < yub):
            raise ValueError(
                "Lower bounds for some dimensions of Y are not smaller than upper bounds"
            )

        self.ylb = ylb
        self.yub = yub
        self._qscale = yub - ylb

    def sample(self, n, seed):
        dx = self.xlb.shape[0]
        dy = self.ylb.shape[0]

        with util.NumpySeedContext(seed=seed):
            X = stats.uniform.rvs(
                loc=self.xlb, scale=self._pscale, size=np.array([n, dx])
            )
            Y = stats.uniform.rvs(
                loc=self.ylb, scale=self._qscale, size=np.array([n, dy])
            )

        label = "ind_unif_dx{}_dy{}".format(dx, dy)
        return PairedData(X, Y, label)

    def dx(self):
        return self.xlb.shape[0]

    def dy(self):
        return self.ylb.shape[0]


class PS2DUnifRotate(PSIndUnif):
    """
    A source where X, Y follow rotated uniform distributions.

    The default uniform distribution is U(-1, 1), rotated by a rotation matrix
    of the specified angle. This simulates the setting of an ICA problem.
    """

    def __init__(self, angle, xlb=-1, xub=1, ylb=-1, yub=1):
        """
        angle: angle in radian
        xlb: lower bound for x (a real number)
        xub: upper bound for x (a real number)
        ylb: lower bound for y (a real number)
        yub: upper bound for y (a real number)
        """
        super().__init__([xlb], [xub], [ylb], [yub])

        self.angle = angle

    def sample(self, n, seed=389):
        t = self.angle
        c, s = np.cos(t), np.sin(t)
        rot = np.array([[c, s], [-s, c]])

        super_pdata = super().sample(n, seed)
        X, Y = super_pdata.xy

        XY = np.hstack((X, Y))
        np.dot(XY, rot, out=XY)

        label = "rot_unif_a{}".format(t)
        return PairedData(XY[:, [0]], XY[:, [1]], label)

    def dx(self):
        return 1

    def dy(self):
        return 1


class PSUnifRotateNoise(PS2DUnifRotate):
    """
    X, Y are dependent in the same way as in PS2DUnifRotate. However, this
    problem adds more extra noise dimensions.
    - The total number of dimensions is 2+2*noise_dim.
    - Only the first dimensions of X and Y are dependent. Dependency strength
        depends on the specified angle.
    """

    def __init__(self, angle, xlb=-1, xub=1, ylb=-1, yub=1, noise_dim=0):
        """
        angle: angle in radian
        xlb: lower bound for x (a real number)
        xub: upper bound for x (a real number)
        ylb: lower bound for y (a real number)
        yub: upper bound for y (a real number)
        noise_dim: number of noise dimensions to add to each of X and Y. All
            the extra dimensions follow U(-1, 1)
        """
        super().__init__(angle, xlb, xub, ylb, yub)
        if noise_dim < 0:
            raise ValueError(
                "Noise dimensions must be a non-negative integer; found {}".format(
                    noise_dim
                )
            )

        self.noise_dim = noise_dim

    def sample(self, n, seed=883):
        super_pdata = super().sample(n, seed)

        noise_dim = self.noise_dim
        new_label = "rot_unif_noisedim{}".format(noise_dim)
        super_pdata.label = new_label

        if self.noise_dim == 0:
            return super_pdata

        with util.NumpySeedContext(seed=seed + 1):
            # Draw points from U(-1, 1)
            Xnoise = stats.uniform.rvs(loc=-1, scale=2, size=(n, noise_dim))
            Ynoise = stats.uniform.rvs(loc=-1, scale=2, size=(n, noise_dim))

        # Concatenate the noise dims to the 2d problem
        X2d, Y2d = super_pdata.xy
        X = np.hstack((X2d, Xnoise))
        Y = np.hstack((Y2d, Ynoise))
        return PairedData(X, Y, label=new_label)

    def dx(self):
        return 1 + self.noise_dim

    def dy(self):
        return 1 + self.noise_dim


class PSSinFreq(PairedSource):
    r"""
    X, Y follow the density proportional to
        1+\prod_{i=1}^{d} [ sin(w*x_i)sin(w*y_i) ]
    w is the frequency. The higher w, the close the density is to a uniform
    distribution on [-pi, pi] x [-pi, pi].
    """

    def __init__(self, freq, d):
        """
        freq: a non-negative floating-point number
        """
        if freq < 0:
            raise ValueError(
                "Frequency must be a non-negative number; found {}".format(freq)
            )

        self.freq = freq
        self.d = d

    @staticmethod
    def sample_d_variates(w, n, D, seed=81, block_size=500):
        """Return an n x D sample matrix using rejection sampling."""
        sam = np.zeros((n, D))
        from_ind = 0
        with util.NumpySeedContext(seed=seed):
            while from_ind < n:
                # Draw x, y from Uniform(-pi, pi).
                # Sample block_size * D at a time.
                X = stats.uniform.rvs(
                    loc=-math.pi, scale=2 * math.pi, size=(block_size, D)
                )

                un_den = 1.0 + np.prod(np.sin(w * X), axis=1)
                unif_draw = stats.uniform.rvs(size=block_size)

                # Reject some samples and select no more than
                # (n - from_ind) samples from the accepted.
                idxs = np.nonzero(unif_draw < un_den / 2.0)[0][: n - from_ind]

                # Aggregate selected to sample
                end_ind = from_ind + len(idxs)
                sam[from_ind:end_ind, :] = X[idxs, :]
                from_ind = end_ind
        return sam

    def sample(self, n, seed=81):
        d = self.d
        sample = self.sample_d_variates(self.freq, n, 2 * d, seed)
        X = sample[:, :d]
        Y = sample[:, d:]
        label = "sin_freq{}_d{}".format(self.freq, d)
        return PairedData(X, Y, label)

    def dx(self):
        return self.d

    def dy(self):
        return self.d


class PS2DSinFreq(PSSinFreq):
    """
    A specialization of PSSinFreq for d == 1.

    This dataset was used in Arthur Gretton's lecture notes.
    """

    def __init__(self, freq):
        """
        freq: a nonnegative floating-point number
        """
        super().__init__(freq, 1)


class PSIndSameGauss(PairedSource):
    """A source returning samples from two standard Gaussians."""

    def __init__(self, dx, dy):
        """
        dx: dimension of X
        dy: dimension of Y
        """
        self._dx = dx
        self._dy = dy

    def sample(self, n, seed=1329):
        with util.NumpySeedContext(seed=seed):
            X = np.random.randn(n, self._dx)
            Y = np.random.randn(n, self._dy)
        label = "sg_dxi{}_dy{}".format(self._dx, self._dy)
        return PairedData(X, Y, label)

    def dx(self):
        return self._dx

    def dy(self):
        return self._dy


class PSPairwiseSign(PairedSource):
    r"""
    A toy problem given in section 5.3 of

    Large-Scale Kernel Methods for Independence Testing
    Qinyi Zhang, Sarah Filippi, Arthur Gretton, Dino Sejdinovic

    X ~ N(0, I_d)
    Y = \sqrt(2/d) \sum_{j=1}^{d/2} sign(X_{2j-1 * X_{2j}})|Z_j| + Z_{d/2+1}

    where Z ~ N(0, I_{d/2+1})
    """

    def __init__(self, dx):
        """
        dx: the dimension of X
        """
        if dx <= 0 or dx % 2 != 0:
            raise ValueError(
                "dx must be a positive even integer; found {}".format(dx)
            )
        self._dx = dx

    def sample(self, n, seed=44):
        d = self._dx
        d_half = d // 2
        with util.NumpySeedContext(seed=seed):
            Z = np.random.randn(n, d_half + 1)
            X = np.random.randn(n, d)

        Y = np.sum(
            np.sign(X[:, 0::2] * X[:, 1::2]) * np.abs(Z[:, :d_half]), axis=1
        )

        Y *= np.sqrt(2.0 / d)
        Y += Z[:, d_half]

        label = "pairwise_sign_dx{}".format(d)
        return PairedData(X, Y[:, np.newaxis], label)

    def dx(self):
        return self._dx

    def dy(self):
        return 1
