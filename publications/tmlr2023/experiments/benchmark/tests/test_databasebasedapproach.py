from approaches import DatabaseWiseApproach
from tqdm import tqdm
import numpy as np

from asforests.cb_computer import EnsemblePerformanceAssessor

from _ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices

from experiments.benchmark.tests.util import ApproachTestClass
from parameterized import parameterized
import unittest

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

# configure logger for tester
approach_logger = logging.getLogger("tested_approach")
approach_logger.handlers.clear()
approach_logger.addHandler(ch)
approach_logger.setLevel(logging.WARNING)

epa_logger = logging.getLogger("tested_approach.epa")
epa_logger.handlers.clear()
epa_logger.addHandler(ch)
epa_logger.setLevel(logging.ERROR)


class TestDatabaseBasedApproach(ApproachTestClass):

    """
        This test checks whether the database-based approach is able to *exactly* determine the true parameters (both conditional and unconditional)

        This is done as follows: We generate a specific sequence of ensemble members, namely each of them exactly once until all have been seen.
        Thereby, the approach has seen all ensemble members exactly once and should, by coincidence, estimate the exactly correct mean values and variances.
    """

    @classmethod
    def setUpClass(cls):
        
        # create set of deviations
        n_samples = 10
        num_different_ensemble_members = 2
        X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=2)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(X, y, train_indices=train_indices, seed=0, num_matrices=num_different_ensemble_members, max_tries=10**2)
        assert num_different_ensemble_members == len(matrices)
        
        # store setup
        cls.matrices = np.array(matrices)
        indices = [classes.index(i) for i in y]
        cls.y_oh = np.eye(len(classes))[indices]
        cls.deviations = matrices - cls.y_oh
        cls.gtc = GroundTruthComputer(deviations=cls.deviations)
    
    def setUp(self):
        self.matrices = self.__class__.matrices
        self.y_oh = self.__class__.y_oh
        self.deviations = self.__class__.deviations
        self.gtc = self.__class__.gtc

        self.t_checkpoints = np.arange(1, 6)
    
    def get_approach(self, seed, estimated_parameters):
        return DatabaseWiseApproach(
            estimated_parameters=estimated_parameters,
            population_mode="stream",
            random_state=seed,
            upper_bound_for_sample_size=10**4,
            logger=approach_logger
        )

    def get_run_approach(self, param):
        a = DatabaseWiseApproach(estimated_parameters=[param], population_mode="stream")
        a.reset()
        a.tell_ground_truth_labels(self.y_oh)

        # tell the approach all possible matrices
        for pm in self.matrices:
            a.receive_predictions_of_new_ensemble_member(pm)
        return a
    
    def test_correct_behavior_of_ensemble_performance_estimator(self):

        epa = EnsemblePerformanceAssessor(upper_bound_for_sample_size=10**8, population_mode="stream")
        
        # add all of these matrices to the estimator
        for m in self.deviations:
            epa.add_deviation_matrix(m)

        # full true deviation table
        deviation_table = self.gtc.get_all_ensemble_data_combinations(t=len(self.deviations), n=2, compute_deviations=True)
        
        # assert correct assessment of means
        estimated_col_means = epa.gap_mean_point
        assert estimated_col_means.shape == (2,)
        for class_index in range(1, 3):
            assert np.isclose(deviation_table[f"D_1{class_index}^1"].mean(), estimated_col_means[class_index-1])

        # assert correct assessment of variances
        estimated_col_vars = epa.gap_var_point
        assert estimated_col_vars.shape == (2,)
        for class_index in range(1, 3):
            assert np.isclose(deviation_table[f"D_1{class_index}^1"].var(ddof=0), estimated_col_vars[class_index-1])

        # assert correct assessment of deviation covariances
        relevant_cols_for_covs = ["x_1", "s_1", "s_2"]
        for class_index in range(1, 3):
            relevant_cols_for_covs.extend([f"D_1{class_index}^1", f"D_1{class_index}^2"])
        relevant_part_of_deviation_table = deviation_table[relevant_cols_for_covs].drop_duplicates().reset_index(drop=True)
        estimated_covs = epa.gap_cov_across_members_point
        assert estimated_covs.shape == (2,)
        for class_index in range(1, 3):
            assert np.isclose(relevant_part_of_deviation_table[[f"D_1{class_index}^1", f"D_1{class_index}^2"]].cov(ddof=0).values[0, 1], estimated_covs[class_index - 1]), f"Wrong covariance estimate for class {class_index}"

    def test_correct_estimation_of_mean_in_iid_setup(self):

        param = "E[Z_nt]"
        
        # run approach
        a = self.get_run_approach(param)
        
        mu_pred_array = a.estimate_performance_mean_in_iid_setup(t=self.t_checkpoints)
        for t, mu_pred_from_array in zip(self.t_checkpoints, mu_pred_array):
            true_mean = self.gtc.get_true_parameter(param=param, t=t)
            pred_mean = a.estimate_performance_mean_in_iid_setup(t=np.array([t]))[0]
            self.assertAlmostEqual(true_mean, pred_mean, msg=f"Final prediciton for E[Z_n,{t}] is not correct.")
            self.assertEqual(mu_pred_from_array, pred_mean)

    def test_correct_estimation_of_mean_in_conditional_setup(self):

        param = "E[Z_nt|D_val]"
        
        # run approach
        a = self.get_run_approach(param)

        # compare predicted (and stored) mean with true mean
        mu_pred_array = a.estimate_performance_mean_in_conditional_setup(t=self.t_checkpoints)
        for t, mu_pred_from_array in zip(self.t_checkpoints, mu_pred_array):
            mu_act = self.gtc.get_true_parameter(param=param, t=t)

            mu_pred = a.estimate_performance_mean_in_conditional_setup(t=np.array([t]))[0]
            self.assertAlmostEqual(mu_act, mu_pred, msg=f"Final variance prediciton for E[Z_n,{t}|D_val] is not correct.")
            self.assertEqual(mu_pred_from_array, mu_pred)

    def test_correct_estimation_of_variances_in_iid_setup(self):

        param = "V[Z_nt]"

        # run approach
        a = self.get_run_approach(param)

        # get extracted covs from approach
        predicted_covs = a.xi_covs_in_iid_setting

        # determine ground truth covariances (iid from the validation data)
        gtt = self.gtc.get_ground_truth_table_under_sample_iid_assumption()
        gt_covariances = self.gtc.get_covariance_terms_for_each_instance_pair(gtt, ddof=0)
        true_covs_as_array = gt_covariances[gt_covariances["i_1"] == 1].drop(columns=["i_1", "i_2"]).values.flatten()

        # compare predictions with ground truth in the 14 covariance terms
        for c, cov in enumerate(true_covs_as_array):
            self.assertAlmostEqual(cov, predicted_covs[c], msg=f"Wrong estimate for case {c}")
        
        # compare predicted (and stored) variance with true variance (on iid samples from the validation data as population)
        n_checkpoints = np.array([2, 3])
        v_pred_array = a.estimate_performance_var_in_iid_setup(n=n_checkpoints, t=self.t_checkpoints)
        for n, var_predictions_for_n in zip(n_checkpoints, v_pred_array):
            for t, v_pred_from_array in zip(self.t_checkpoints, var_predictions_for_n):
                true_var = self.gtc.get_true_parameter(param=param, n=n, t=t)
                pred_var = a.estimate_performance_var_in_iid_setup(n=np.array([n]), t=np.array([t]))[0, 0]
                self.assertAlmostEqual(true_var, pred_var, msg=f"Final variance prediciton for V[Z_2,{t}] is not correct, but the covariance estimates are. So this is a matter of aggregation.")
                self.assertEqual(v_pred_from_array, pred_var)

    def test_correct_estimation_of_variance_in_conditional_setup(self):

        param = "V[Z_nt|D_val]"

        a = self.get_run_approach(param)

        # get extracted covs from approach
        predicted_covs = a.xi_covs_in_conditional_setting

        # determine ground truth covariances
        gtt = self.gtc.get_conditional_ground_truth_table()
        gt_covariances = self.gtc.get_covariance_terms_for_each_instance_pair(gtt, ddof=0)
        
        # compare predictions with ground truth in the n x n x 7 covariance terms
        for (i1, i2), df_covariances in gt_covariances.groupby(["i_1", "i_2"]):
            self.assertEqual(1, len(df_covariances))
            covs = df_covariances.drop(columns=["i_1", "i_2"]).iloc[0].values
            for c, cov in enumerate(covs):
                self.assertAlmostEqual(cov, predicted_covs[i1 - 1, i2 - 1, c], msg=f"Wrong estimate for case {c} on instance pair {i1}/{i2}")
        
        # compare predicted (and stored) variance with true variance, and make sure that result is the same as in vectorized one
        v_pred_array = a.estimate_performance_var_in_conditional_setup(t=self.t_checkpoints)
        for t, v_pred_from_array in zip(self.t_checkpoints, v_pred_array):
            v_act = self.gtc.get_true_parameter(param=param, t=t)
            v_pred = a.estimate_performance_var_in_conditional_setup(t=np.array([t]))[0]
            self.assertAlmostEqual(v_act, v_pred, msg=f"Final variance prediciton for V[Z_2,{t}|D_val] is not correct, but the covariance estimates are. So this is a matter of aggregation.")
            self.assertEqual(v_pred_from_array, v_pred)

    
    def test_approximation_quality_for_variance_in_iid_setting(self):
        return

        param = "V[Z_nt]"
      
        # create set of deviations
        n_samples = 200
        n_classes = 2
        num_different_ensemble_members = 3
        X, y = make_classification(n_classes=n_classes, n_samples=n_samples, n_features=20, random_state=0)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=20, random_state=0).split(X, y))
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

        # create both algorithms, the one with enough space and the approximator
        gt_computer = DatabaseWiseApproach(estimated_parameters=[param], population_mode="stream", logger=logger)
        approximator = DatabaseWiseApproach(
            estimated_parameters=[param],
            population_mode="stream",
            upper_bound_for_sample_size=10**6,
            logger=logger)
        for a in [gt_computer, approximator]:
            a.reset()
            a.tell_ground_truth_labels(y_oh)
        
        # advance both and check the approximation quality of the covariance terms
        coeffiecients_for_t10 = gt_computer.get_xi_cov_coefficients_for_iid_scenario(t=np.array([10])).reshape(2, 7) / 2 # divide by 2 due to n = 2, applies for both rows in this special case
        for r, pm in enumerate(matrices, start=1):
            logger.info(f"\n{''.join(['-']*20)}\nStart of round {r}\n{''.join(['-']*20)}")
            gt_computer.receive_predictions_of_new_ensemble_member(pm)
            approximator.receive_predictions_of_new_ensemble_member(pm)
            for outer_index in range(2):
                for inner_index in range(7):
                    b1 = gt_computer.epa.mixed_moment_builders_for_iid_xi_covs[outer_index, inner_index]
                    b2 = approximator.epa.mixed_moment_builders_for_iid_xi_covs[outer_index, inner_index]
                    c1 = coeffiecients_for_t10[outer_index, inner_index] * b1.cov
                    c2 = coeffiecients_for_t10[outer_index, inner_index] * b2.cov
                    logger.info(f"{outer_index}, {inner_index} -> {b1.cov}, {b2.cov} -> {c1}, {c2}")
                    if np.round(c1, 5) != np.round(c2, 5):
                        logger.warning(
                            f"Approximation warning\n"
                            f"\tApproximation of cov for case [{outer_index}, {inner_index}] is wrong by {abs(b1.cov - b2.cov)}.\n"
                            f"\tExpected {b1.cov} but saw {b2.cov}\n"
                            f"\tThe summand then is then not the expected {c1} but {c2}"
                        )
                    #self.assertAlmostEqual(b1.cov, b2.cov, places=5, msg=f"Approximation of cov for case [{outer_index}, {inner_index}] is bad")


        # check that variance is similar
        self.assertAlmostEqual(
            gt_computer.estimate_performance_var_for_two_instances_in_iid_setup(t=np.array([10]))[0],
            approximator.estimate_performance_var_for_two_instances_in_iid_setup(t=np.array([10]))[0],
            places=3
        )