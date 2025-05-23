import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from approaches import DatabaseWiseApproach
from tqdm import tqdm
import numpy as np

from _ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices

from unittest import TestCase
from parameterized import parameterized

import itertools as it

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


class TestDatabaseBasedApproach(TestCase):

    def test_approximation_correctness(self):

        # get deviations
        num_different_ensemble_members = 2
        n_samples = 4
        X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=2)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=2, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(X, y, train_indices=train_indices, seed=0, num_matrices=num_different_ensemble_members, max_tries=10**2)
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
        approx_mean, approx_var = gt.approximate_true_parameters_in_iid_setting_by_sampling(t_checkpoints=t, n_checkpoints=n, num_samples=10**6, num_samples_per_job=(10**6)//5, n_jobs=1)
        t_end = time()
        print(f"finished after {t_end - t_start}s")
        true_std = np.sqrt(true_var)
        approx_std = np.sqrt(approx_var)
        print(f"True std is {true_std}")
        print(f"Approximated std is {approx_std}")
        self.assertAlmostEqual(true_mean, approx_mean, places=4)
        self.assertAlmostEqual(true_std, approx_std, places=4)


if __name__ == "__main__":


    t = 10
    n = 2
    
     # get deviations
    num_different_ensemble_members = 4
    n_samples = 100
    X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=2)
    train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=2, random_state=0).split(X, y))
    matrices, classes = get_unique_prediction_matrices(X, y, train_indices=train_indices, seed=0, num_matrices=num_different_ensemble_members, max_tries=10**2)
    assert num_different_ensemble_members == len(matrices)
    matrices = np.array(matrices)
    indices = [classes.index(i) for i in y]
    y_oh = np.eye(len(classes))[indices]
    deviations = matrices - y_oh

    gtc = GroundTruthComputer(deviations=deviations)
    #print(gtc.get_all_ensemble_data_combinations(n=2, t=10)["z"].var())
    #print(gtc.get_all_ensemble_data_combinations(n=3, t=10)["z"].var() * (3/2)**2)
    #print(gtc.get_all_ensemble_data_combinations(n=2, t=t)["z"].mean())
    print(gtc.get_all_ensemble_data_combinations(n=1, t=t)["z"].mean())


    # compute ground truth with database approach
    dba = DatabaseWiseApproach(estimated_parameters=["E[Z_nt]", "V[Z_nt]"], upper_bound_for_sample_size=10**9, logger=logger)
    dba.reset()
    dba.tell_ground_truth_labels(y_oh=y_oh)
    for m in matrices:
        dba.receive_predictions_of_new_ensemble_member(m)
    true_mean = dba.estimate_performance_mean_in_iid_setup(t=np.array(t))
    true_var = dba.estimate_performance_var_for_two_instances_in_iid_setup(t=np.array([t]))
    print(dba.epa.data_points_processed_for_cov_estimate)
    print(f"True mean: {true_mean}")
    print(f"True std: {np.sqrt(true_var)}")

    # approximate ground truth
    gt = GroundTruthComputer(deviations=deviations)

    num_samples = 10**9

    # approximate ground truth
    for n_jobs in [16, 32, 64, 128]:
        t_start = time()
        approx_mean, approx_var = gt.approximate_true_parameters_in_iid_setting_by_sampling(t_checkpoints=t, n_checkpoints=n, num_samples=num_samples, num_samples_per_job=int(np.ceil(num_samples / n_jobs)), n_jobs=n_jobs)
        t_end = time()
        print(f"finished after {t_end - t_start}s using {n_jobs} CPUs")
        true_std = np.sqrt(true_var)
        approx_std = np.sqrt(approx_var)
        print(f"True mean is {true_mean}")
        print(f"Approximated mean is {approx_mean}")
        print(f"True std is {true_std}")
        print(f"Approximated std is {approx_std}")
