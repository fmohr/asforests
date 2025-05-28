import numpy as np

from asforests.momenter_ import Momenter, MixedMomentBuilder

from unittest import TestCase
from parameterized import parameterized

import logging


# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.INFO)

# configure logger for benchmark
bm_logger = logging.getLogger("benchmark")
bm_logger.handlers.clear()
bm_logger.addHandler(ch)
bm_logger.setLevel(logging.WARN)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class TestMomenters(TestCase):

    def test_mean_estimates(self):

        rs = np.random.RandomState(0)
        matrices = [rs.rand(10, 3)]
        
        momenter = Momenter()
        for m in matrices:
            momenter.add_batch(m)
        assert momenter.means.shape == (3,)
        assert np.all(np.mean(np.concatenate(matrices, axis=0), axis=0) == momenter.means)

    def test_variance_estimates(self):

        rs = np.random.RandomState(0)
        matrices = [rs.rand(10, 3)]
        
        momenter = Momenter()
        for m in matrices:
            momenter.add_batch(m)
        assert momenter.means.shape == (3,)
        assert np.all(np.isclose(np.var(np.concatenate(matrices, axis=0), axis=0, ddof=0), momenter.central_moments[1]))

    @parameterized.expand([
        True, False
    ])
    def test_covariance_estimates(self, bias):

        rs = np.random.RandomState(0)

        # first check the univariate case (only one covariance simultaneously)
        vectors_1 = [rs.rand(10)]
        vectors_2 = [rs.rand(10)]
        momenter = MixedMomentBuilder(biased_covariance_estimate=bias)
        for v1, v2 in zip(vectors_1, vectors_2):
            momenter.add_observations(v1, v2)
        assert type(momenter.cov) == np.float64
        true_cov = np.cov(np.concatenate(vectors_1, axis=0), np.concatenate(vectors_2, axis=0), bias=bias)[0, 1]
        assert np.isclose(true_cov, momenter.cov)
        
        # now check that covariances are also correctly estimated if we estimate several at a time
        matrices_1 = [rs.rand(10, 3)]
        matrices_2 = [rs.rand(10, 3)]
        matrix_1 = np.concatenate(matrices_1, axis=0).T
        matrix_2 = np.concatenate(matrices_2, axis=0).T
        true_covs = np.array([
            np.cov(vector_1, vector_2, bias=bias)[0, 1]
            for vector_1, vector_2 in zip(matrix_1, matrix_2)
        ])

        # first check whether independent instances of the MixedMomentBuilder get the covariances right
        print("Individual")
        for true_cov, vector_1, vector_2 in zip(true_covs, matrix_1, matrix_2):
            momenter = MixedMomentBuilder(biased_covariance_estimate=bias)
            momenter.add_observations(vector_1, vector_2)
            assert np.isclose(true_cov, momenter.cov)

        # now check whether a simultaneous computation of the covariances works
        print("Group")
        momenter = MixedMomentBuilder(biased_covariance_estimate=bias)
        for m1, m2 in zip(matrices_1, matrices_2):
            momenter.add_observations(m1, m2)
        assert momenter.cov.shape == (3,)
        assert np.all(np.isclose(true_covs, momenter.cov)), f"Covariance estimates seem to work for univariate cases (only one pair of variables at a time), but not for multiple variable pairs when bias is set to {bias}"