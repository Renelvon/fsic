"""
Module for testing feature module.
"""

__author__ = "wittawat"

import unittest

import numpy as np

from fsic import feature, kernel, util


class TestMarginalCDFMap(unittest.TestCase):
    def test_general(self):
        n = 30
        d = 4
        X = np.random.randn(n, d) * 3 + 4

        M = feature.MarginalCDFMap()
        Z = M.gen_features(X)

        # assert
        self.assertEqual(Z.shape[1], d)
        self.assertEqual(Z.shape[0], n)
        self.assertEqual(M.num_features(X), d)
        self.assertTrue(np.all(Z >= 0))
        self.assertTrue(np.all(Z <= 1))


class TestRFFKGauss(unittest.TestCase):
    def test_general(self):
        n = 31
        d = 3
        X = np.random.rand(n, d) * 2 - 4

        sigma2 = 3.7
        feature_pairs = 51
        rff = feature.RFFKGauss(sigma2, feature_pairs, seed=2)
        Z = rff.gen_features(X)
        Z2 = rff.gen_features(X)

        # assert sizes
        self.assertEqual(Z.shape[0], n)
        self.assertEqual(Z.shape[1], 2 * feature_pairs)

        # assert deterministicity
        np.testing.assert_array_almost_equal(Z, Z2)

    def test_approximation(self):
        n = 100
        d = 3
        X = np.random.rand(n, d) * 2 - 4

        sigma2 = 2.7
        feature_pairs = 50
        rff = feature.RFFKGauss(sigma2, feature_pairs, seed=2)
        Z = rff.gen_features(X)
        Krff = Z.dot(Z.T)

        # check approximation quality
        k = kernel.KGauss(sigma2)
        K = k.eval(X, X)
        diff = np.linalg.norm((Krff - K), "fro")
        self.assertLessEqual(diff / n ** 2, 0.5)

        # print 'fro diff: %.3f'%np.linalg.norm( (Krff - K)/n**2, 'fro')


class TestNystromFeatureMap(unittest.TestCase):
    def test_approximation(self):
        for s in [298, 67]:
            with util.NumpySeedContext(seed=s):
                k = kernel.KGauss(1)
                n = 50
                d = 3
                X = np.random.randn(n, d) * 3 + 5
                D = n // 3
                induce = util.subsample_rows(X, D, seed=s + 1)
                nymap = feature.NystromFeatureMap(k, induce)

                K = k.eval(X, X)
                Z = nymap.gen_features(X)

                # check approximation quality
                diff = np.linalg.norm((K - Z.dot(Z.T)), "fro")
                self.assertLessEqual(diff / n ** 2, 0.5)

                # check sizes
                self.assertEqual(Z.shape[1], D)
                self.assertEqual(Z.shape[0], n)
