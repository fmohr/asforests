from approaches import DatabaseWiseApproach, BootstrappingApproach
from tqdm import tqdm
import numpy as np

from asforests.cb_computer import EnsemblePerformanceAssessor

from _ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices

from unittest import TestCase
from parameterized import parameterized

import itertools as it

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


class TestBootstrappingApproach(TestCase):

    def test_approximation_quality_for_variance_in_iid_setting(self):

        param = "V[Z_nt]"
      
        # create set of deviations
        n_samples = 20
        n_classes = 2
        num_different_ensemble_members = 5
        X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=20, random_state=0)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=10, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(
            X,
            y,
            train_indices=train_indices,
            seed=0,
            num_matrices=num_different_ensemble_members,
            max_tries=10**2,
            rf_kwargs={"max_depth": 1}
        )
        matrices = np.array(matrices)
        assert (num_different_ensemble_members, n_samples, n_classes) == matrices.shape

        # store setup
        indices = [classes.index(i) for i in y]
        y_oh = np.eye(len(classes))[indices]
        deviations = matrices - y_oh

        t_checkpoints = np.array([10, 100, 1000])

        # determine ground truth variance
        gt_computer = DatabaseWiseApproach(
            estimated_parameters=[param],
            population_mode="stream",
            logger=logger
        )
        gt_computer.reset()
        gt_computer.tell_ground_truth_labels(y_oh=y_oh)
        for pm in matrices:
            gt_computer.receive_predictions_of_new_ensemble_member(pm)
        true_stds = np.sqrt(gt_computer.estimate_performance_var_for_two_instances_in_iid_setup(t_checkpoints))
        
        # see whether we can approximate this with bootstrapping
        ba = BootstrappingApproach(bootstrap_size=10**1, num_resamples=10**3, logger=logger)
        ba.reset()
        ba.tell_ground_truth_labels(y_oh=y_oh)
        for pm in matrices:
            ba.receive_predictions_of_new_ensemble_member(pm)
        estimated_stds = np.sqrt(ba.estimate_performance_var_for_two_instances_in_iid_setup(t_checkpoints))
        logger.info(f"True vs estimated std at t={t_checkpoints}: {true_stds} vs {estimated_stds}")
        for t, v_true, v_est in zip(t_checkpoints, true_stds, estimated_stds):
            self.assertAlmostEqual(v_true, v_est)
    
    def test_correct_estimation_of_variance_in_conditional_setup(self):

        param = "V[Z_nt|D_val]"
        return

        # create set of deviations
        n_samples = 20
        n_classes = 2
        num_different_ensemble_members = 5
        X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=20, random_state=0)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=4, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(
            X,
            y,
            train_indices=train_indices,
            seed=0,
            num_matrices=num_different_ensemble_members,
            max_tries=10**2,
            rf_kwargs={"max_depth": 1}
        )
        matrices = np.array(matrices)
        assert (num_different_ensemble_members, n_samples, n_classes) == matrices.shape

        # store setup
        indices = [classes.index(i) for i in y]
        y_oh = np.eye(len(classes))[indices]
        deviations = matrices - y_oh

        # determine ground truth variance
        gt_computer = DatabaseWiseApproach(
            estimated_parameters=[param],
            population_mode="stream",
            logger=logger
        )
        gt_computer.reset()
        gt_computer.tell_ground_truth_labels(y_oh=y_oh)
        for pm in matrices:
            gt_computer.receive_predictions_of_new_ensemble_member(pm)
        true_variance_at_t10 = gt_computer.estimate_performance_var_in_conditional_setup(np.array([10]))[0]
        
        # see whether we can approximate this with bootstrapping
        ba = BootstrappingApproach(bootstrap_size=10**1, num_resamples=10**1, logger=logger)
        ba.reset()
        ba.tell_ground_truth_labels(y_oh=y_oh)
        print(matrices)
        for pm in matrices:
            ba.receive_predictions_of_new_ensemble_member(pm)
        estimated_variance_at_t10 = ba.estimate_performance_var_in_conditional_setup(np.array([10]))[0]
        self.assertAlmostEqual(true_variance_at_t10, estimated_variance_at_t10)