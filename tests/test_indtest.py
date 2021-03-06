"""
Module for testing indtest module.
"""

import unittest

import numpy as np

from fsic import data, feature, indtest, kernel, util


def get_pdata_mean(n, dx=2):
    X = np.random.randn(n, dx)
    Y = np.mean(X, 1)[:, np.newaxis] + np.random.randn(n, 1) * 0.01
    return data.PairedData(X, Y, label="mean")


def kl_median(pdata):
    """
    Get two Gaussian kernels constructed with the median heuristic.
    Randomize V, W from the standard Gaussian distribution.
    """
    xtr, ytr = pdata.xy()
    medx2 = util.median_distance(xtr) ** 2
    medy2 = util.median_distance(ytr) ** 2
    k = kernel.KGauss(medx2)
    l = kernel.KGauss(medy2)
    return k, l


class TestNFSIC(unittest.TestCase):
    def setUp(self):
        n = 300
        dx = 2
        pdata_mean = get_pdata_mean(n, dx)
        X, Y = pdata_mean.xy()
        gwx2 = util.median_distance(X) ** 2
        gwy2 = util.median_distance(Y) ** 2
        k = kernel.KGauss(gwx2)
        l = kernel.KGauss(gwy2)
        J = 2
        V = np.random.randn(J, dx)
        W = np.random.randn(J, 1)

        self.nfsic = indtest.NFSIC(k, l, V, W, alpha=0.01)
        self.pdata_mean = pdata_mean

    @unittest.skip("Should reject. Cannot assert this for sure.")
    def test_perform_test(self):
        test_result = self.nfsic.perform_test(self.pdata_mean)
        self.assertTrue(test_result["h0_rejected"], "Test should reject H0")

    def test_compute_stat(self):
        stat = self.nfsic.compute_stat(self.pdata_mean)
        self.assertGreater(stat, 0)

    def test_list_permute(self):
        # Check that the relative frequency in the simulated histogram is
        # accurate enough.
        ps = data.PS2DSinFreq(freq=2)
        n_permute = 1000
        J = 4
        for s in [284, 77]:
            with util.NumpySeedContext(seed=s):
                pdata = ps.sample(n=200, seed=s + 1)
                dx = pdata.dx()
                dy = pdata.dy()
                X, Y = pdata.xy()

                k = kernel.KGauss(2)
                l = kernel.KGauss(3)
                V = np.random.randn(J, dx)
                W = np.random.randn(J, dy)
                # nfsic = indtest.NFSIC(k, l, V, W, alpha=0.01, reg=0, n_permute=n_permute,
                #        seed=s+3):

                # nfsic_result = nfsic.perform_test(pdata)
                arr = indtest.NFSIC.list_permute(
                    X, Y, k, l, V, W, n_permute=n_permute, seed=s + 34, reg=0
                )
                arr_naive = indtest.NFSIC._list_permute_naive(
                    X, Y, k, l, V, W, n_permute=n_permute, seed=s + 389, reg=0
                )

                # make sure that the relative frequency of the histogram does
                # not differ much.
                freq_a, _ = np.histogram(arr)
                freq_n, _ = np.histogram(arr_naive)
                nfreq_a = freq_a / np.sum(freq_a)
                nfreq_n = freq_n / np.sum(freq_n)
                arr_diff = np.abs(nfreq_a - nfreq_n)
                self.assertTrue(np.all(arr_diff <= 0.2))


class TestGaussNFSIC(unittest.TestCase):
    def setUp(self):
        n = 300
        dx = 2
        pdata_mean = get_pdata_mean(n, dx)
        X, Y = pdata_mean.xy()
        gwx2 = util.median_distance(X) ** 2
        gwy2 = util.median_distance(Y) ** 2
        J = 2
        V = np.random.randn(J, dx)
        W = np.random.randn(J, 1)

        self.gnfsic = indtest.GaussNFSIC(gwx2, gwy2, V, W, alpha=0.01)
        self.pdata_mean = pdata_mean

    @unittest.skip("Should reject. Cannot assert this for sure.")
    def test_perform_test(self):
        test_result = self.gnfsic.perform_test(self.pdata_mean)
        self.assertTrue(test_result["h0_rejected"], "Test should reject H0")

    def test_compute_stat(self):
        stat = self.gnfsic.compute_stat(self.pdata_mean)
        self.assertGreater(stat, 0)


class TestQuadHSIC(unittest.TestCase):
    def setUp(self):
        n = 50
        dx = 2
        pdata_mean = get_pdata_mean(n, dx)
        k, l = kl_median(pdata_mean)

        self.qhsic = indtest.QuadHSIC(k, l, n_permute=60, alpha=0.01)
        self.pdata_mean = pdata_mean

    def test_list_permute(self):
        # test that the permutations are done correctly.
        # Test against a naive implementation.
        pd = self.pdata_mean
        X, Y = pd.xy()
        k = self.qhsic.k
        l = self.qhsic.l
        n_permute = self.qhsic.n_permute
        s = 113
        arr_hsic = indtest.QuadHSIC.list_permute(X, Y, k, l, n_permute, seed=s)
        arr_hsic_naive = indtest.QuadHSIC._list_permute_generic(
            X, Y, k, l, n_permute, seed=s
        )
        np.testing.assert_array_almost_equal(arr_hsic, arr_hsic_naive)
        # 'Permuted HSIC values are not the same as the naive implementation.')


class TestFiniteFeatureHSIC(unittest.TestCase):
    def test_list_permute_spectral(self):
        # make sure that simulating from the spectral approach is roughly the
        # same as doing permutations.
        ps = data.PS2DSinFreq(freq=2)
        n_features = 5
        n_simulate = 3000
        n_permute = 3000
        for s in [283, 2]:
            with util.NumpySeedContext(seed=s):
                pdata = ps.sample(n=200, seed=s + 1)
                X, Y = pdata.xy()

                sigmax2 = 1
                sigmay2 = 0.8
                fmx = feature.RFFKGauss(
                    sigmax2, n_features=n_features, seed=s + 3
                )
                fmy = feature.RFFKGauss(
                    sigmay2, n_features=n_features, seed=s + 23
                )

                Zx = fmx.gen_features(X)
                Zy = fmy.gen_features(Y)
                list_perm = indtest.FiniteFeatureHSIC.list_permute(
                    X, Y, fmx, fmy, n_permute=n_permute, seed=s + 82
                )
                (
                    list_spectral,
                    _,
                    _,
                ) = indtest.FiniteFeatureHSIC.list_permute_spectral(
                    Zx, Zy, n_simulate=n_simulate, seed=s + 119
                )

                # make sure that the relative frequency of the histogram does
                # not differ much.
                freq_p, _ = np.histogram(list_perm)
                freq_s, _ = np.histogram(list_spectral)
                nfreq_p = freq_p / np.sum(freq_p)
                nfreq_s = freq_s / np.sum(freq_s)
                arr_diff = np.abs(nfreq_p - nfreq_s)
                self.assertTrue(np.all(arr_diff <= 0.2))


class TestRDC(unittest.TestCase):
    def test_rdc(self):
        feature_pairs = 10
        n = 30
        for f in range(1, 7):
            ps = data.PS2DSinFreq(freq=1)
            pdata = ps.sample(n, seed=f + 4)
            fmx = feature.RFFKGauss(1, feature_pairs, seed=f + 10)
            fmy = feature.RFFKGauss(2.0, feature_pairs + 1, seed=f + 9)
            rdc = indtest.RDC(fmx, fmy, alpha=0.01)
            stat, evals = rdc.compute_stat_with_eigvals(pdata)

            self.assertGreaterEqual(stat, 0)
            abs_evals = np.abs(evals)
            self.assertTrue(np.all(abs_evals >= 0))
            self.assertTrue(np.all(abs_evals <= 1))


class TestFuncs(unittest.TestCase):
    """
    This is to test functions that do not belong to any class.
    """

    def test_nfsic(self):
        n = 50
        dx = 3
        dy = 1
        X = np.random.randn(n, dx)
        Y = np.random.randn(n, dy) + 1
        medx2 = util.median_distance(X) ** 2
        medy2 = util.median_distance(Y) ** 2
        k = kernel.KGauss(medx2)
        l = kernel.KGauss(medy2)
        J = 3
        V = np.random.randn(J, dx)
        W = np.random.randn(J, dy)

        nfsic, _, _ = indtest.nfsic(X, Y, k, l, V, W, reg=0)

        self.assertAlmostEqual(np.imag(nfsic), 0)
        self.assertGreater(nfsic, 0)


if __name__ == "__main__":
    unittest.main()
