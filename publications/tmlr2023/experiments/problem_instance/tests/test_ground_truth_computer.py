import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

from experiments.problem_instance._ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices

from unittest import TestCase
from parameterized import parameterized


from time import time

import logging



# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


class TestGroundTruthComputer(TestCase):

    @parameterized.expand(range(10))
    def test_approximation_correctness_for_iid(self, seed):

        # get deviations
        num_different_ensemble_members = 2
        n_samples = 4
        X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=seed)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=2, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(X, y, train_indices=train_indices, seed=seed, num_matrices=num_different_ensemble_members, max_tries=10**2)
        assert num_different_ensemble_members == len(matrices)
        matrices = np.array(matrices)
        indices = [classes.index(i) for i in y]
        y_oh = np.eye(len(classes))[indices]
        deviations = matrices - y_oh

        # get ground truth computer
        gt = GroundTruthComputer(deviations=deviations)

        # approximate ground truth
        t = 10
        n = 3
        true_var = gt.get_true_parameter("V[Z_nt]", t=t, n=n)
        if true_var == 0:
            raise Exception(f"No gag in estimating variance of 0")
        print(f"True var is {true_var}")
        true_mean = gt.get_true_parameter("E[Z_nt]", t=t)

        # approximate ground truth
        t_start = time()
        approx_mean, approx_var = gt.approximate_true_parameters_in_iid_setting_by_sampling(t_checkpoints=t, n_checkpoints=n, num_samples=10**8, num_samples_per_job=2*10**5, n_jobs=1)
        t_end = time()
        print(f"finished after {t_end - t_start}s")
        approx_mean = approx_mean[0, 0]
        approx_var = approx_var[0, 0]
        true_std = np.sqrt(true_var)
        approx_std = np.sqrt(approx_var)
        print(f"True std is {true_std}")
        print(f"Approximated mean is {approx_mean} and approximated std is {approx_std}")
        self.assertAlmostEqual(true_mean, approx_mean, places=4, msg=f"E[Z_nt] not correctly approximated. True value is {true_mean}, approximated value is {approx_mean}")
        self.assertAlmostEqual(true_std, approx_std, places=4, msg=f"sqrt(V[Z_nt]) not correctly approximated. True value is {true_std}, approximated value is {approx_std}")

    @parameterized.expand(range(10))
    def test_approximation_correctness_for_cond(self, seed):

        # get deviations
        num_different_ensemble_members = 2
        n_samples = 4
        X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=seed)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=2, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(X, y, train_indices=train_indices, seed=seed, num_matrices=num_different_ensemble_members, max_tries=10**2)
        assert num_different_ensemble_members == len(matrices)
        matrices = np.array(matrices)
        indices = [classes.index(i) for i in y]
        y_oh = np.eye(len(classes))[indices]
        deviations = matrices - y_oh

        # get ground truth computer
        gt = GroundTruthComputer(deviations=deviations)

        # approximate ground truth
        t = 10
        true_var = gt.get_true_parameter("V[Z_nt|D_val]", t=t)
        if true_var == 0:
            raise Exception(f"No gag in estimating variance of 0")
        print(f"True var is {true_var}")
        true_mean = gt.get_true_parameter("E[Z_nt|D_val]", t=t)

        # approximate ground truth
        t_start = time()
        approx_mean, approx_var = gt.approximate_true_parameters_in_cond_setting_by_sampling(t_checkpoints=t, num_samples=10**8, num_samples_per_job=2*10**5, n_jobs=1)
        t_end = time()
        print(f"finished after {t_end - t_start}s")
        approx_mean = approx_mean[0]
        approx_var = approx_var[0]
        true_std = np.sqrt(true_var)
        approx_std = np.sqrt(approx_var)
        print(f"True std is {true_std}")
        print(f"Approximated mean is {approx_mean} and approximated std is {approx_std}")
        self.assertAlmostEqual(true_mean, approx_mean, places=4, msg=f"E[Z_nt|D_val] not correctly approximated. True value is {true_mean}, approximated value is {approx_mean}")
        self.assertAlmostEqual(true_std, approx_std, places=4, msg=f"sqrt(V[Z_nt|D_val]) not correctly approximated. True value is {true_std}, approximated value is {approx_std}")
