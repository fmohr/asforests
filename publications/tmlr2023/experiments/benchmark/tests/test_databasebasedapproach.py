from approaches import DatabaseWiseApproach
import numpy as np

from asforests.cb_computer import EnsemblePerformanceAssessor

from experiments.problem_instance._ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices

from experiments.benchmark.tests.util import ApproachTestClass, ProblemInstanceWrapperForTesting

import time

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
approach_logger.setLevel(logging.DEBUG)

epa_logger = logging.getLogger("tested_approach.epa")
epa_logger.handlers.clear()
#epa_logger.addHandler(ch)
epa_logger.setLevel(logging.DEBUG)


def create_case(n_samples=10, num_different_ensemble_members=2):
        
        # create set of deviations
        X, y = make_classification(n_classes=2, n_samples=n_samples, n_features=20, random_state=2)
        train_indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=0).split(X, y))
        matrices, classes = get_unique_prediction_matrices(X, y, train_indices=train_indices, seed=0, num_matrices=num_different_ensemble_members, max_tries=10**2)
        assert num_different_ensemble_members == len(matrices)
        
        # store setup
        matrices = np.array(matrices)
        indices = [classes.index(i) for i in y]
        y_oh = np.eye(len(classes))[indices]
        deviations = matrices - y_oh
        return matrices, y_oh, deviations

class TestDatabaseBasedApproach(ApproachTestClass):

    """
        This test checks whether the database-based approach is able to *exactly* determine the true parameters (both conditional and unconditional)

        This is done as follows: We generate a specific sequence of ensemble members, namely each of them exactly once until all have been seen.
        Thereby, the approach has seen all ensemble members exactly once and should, by coincidence, estimate the exactly correct mean values and variances.
    """    
    def setUp(self, n_samples=10, num_different_ensemble_members=5):
        self.matrices, self.y_oh, self.deviations = create_case(n_samples=n_samples, num_different_ensemble_members=num_different_ensemble_members)
        self.gtc = GroundTruthComputer(deviations=self.deviations)

        self.t_checkpoints = np.arange(1, 6)
    
    def get_approach(self, seed, estimated_parameters):
        return DatabaseWiseApproach(
            estimated_parameters=estimated_parameters,
            population_mode="stream",
            random_state=seed,
            threshold_for_number_of_samples_to_exclude_param=10**6,
            logger=approach_logger
        )

    def get_run_approach(self, param, threshold_for_number_of_samples_to_exclude_param=10**4):
        a = DatabaseWiseApproach(
            estimated_parameters=[param],
            population_mode="stream",
            threshold_for_number_of_samples_to_exclude_param=threshold_for_number_of_samples_to_exclude_param,
            logger=approach_logger
        )
        a.reset()
        a.tell_ground_truth_labels(self.y_oh)

        # tell the approach all possible matrices
        round = 0
        print(f"Starting run with a total of {len(self.matrices)} matrices.")
        for pm in self.matrices:
            round += 1
            print(f"Starting round #{round}")
            a.receive_predictions_of_new_ensemble_member(pm)
            print(f"Finished round #{round}")
        return a
    
    def test_callbacks(self):

        from asforests.cb_computer import Callback
        import pandas as pd

        class MyCallback(Callback):
            
            def __init__(self):
                super().__init__()
                self.cnt_start = 0
                self.cnt_end = 0
                self.xi_pairs_list = []
            
            def on_round_start(self):
                self.cnt_start += 1
            
            def on_xi_term_pair_computation(self, cov_updater):
                if len(self.xi_pairs_list) == cov_updater.finished_rounds:
                    self.xi_pairs_list.append(cov_updater.new_xi_pairs)

            def on_round_end(self):
                self.cnt_end += 1

        cb = MyCallback()

        a = DatabaseWiseApproach(
            population_mode="stream",
            random_state=0,
            threshold_for_number_of_samples_to_exclude_param=10**6,
            callbacks=[cb],
            logger=approach_logger
        )
        
        pi = ProblemInstanceWrapperForTesting(
            ensemble_seed=0,
            num_possible_ensemble_members=8
        )
        a.reset()
        a.tell_ground_truth_labels(pi.pi.y_oh_val)
        for pm in pi.pi.predictions_val:
            a.receive_predictions_of_new_ensemble_member(pm)
        
        self.assertEqual(len(pi.pi.predictions_val), cb.cnt_start)
        self.assertEqual(len(pi.pi.predictions_val), cb.cnt_end)
        self.assertEqual(len(pi.pi.predictions_val), len(cb.xi_pairs_list))

    def test_that_cov_updaters_are_disabled_if_no_variances_are_estimated(self):
        for param in ["V[Z_nt]", "V[Z_nt|D_val]"]:
            a = DatabaseWiseApproach(
                estimated_parameters=[param],
                population_mode="stream",
                random_state=0,
                logger=approach_logger
            )

            pi = ProblemInstanceWrapperForTesting(
                ensemble_seed=0,
                num_possible_ensemble_members=8,
                validation_size=5
            )
            a.reset()
            a.tell_ground_truth_labels(pi.pi.y_oh_val)
            round = 0

            for pm in pi.pi.get_prediction_matrix_generator(ensemble_sequence_seed=0, only_validation_data=True):
                a.receive_predictions_of_new_ensemble_member(pm)
                if param == "V[Z_nt]":
                    self.assertIsNone(a.epa.cov_updater_for_conditional_case)
                elif param == "V[Z_nt|D_val]":
                    self.assertIsNone(a.epa.cov_updater_for_iid_case_equal_instances)
                    self.assertIsNone(a.epa.cov_updater_for_iid_case_arbitrary_instances)
                round += 1
                if round > 5:
                    break
    
    def test_that_cov_updaters_increase_samples_for_active_params(self):
        position_wise_thresholds = 10**3#np.array([10**3, 10**3, 10**3, 10**3, 10**4, 10**4, 10**5])
        a = DatabaseWiseApproach(
            population_mode="stream",
            random_state=0,
            threshold_for_number_of_samples_to_exclude_param=position_wise_thresholds,
            logger=approach_logger
        )

        pi = ProblemInstanceWrapperForTesting(
            ensemble_seed=0,
            num_possible_ensemble_members=8,
            validation_size=5
        )
        a.reset()
        a.tell_ground_truth_labels(pi.pi.y_oh_val)
        round = 0

        updaters = None
        last_count_of_observations = None

        for pm in pi.pi.get_prediction_matrix_generator(ensemble_sequence_seed=0, only_validation_data=True):
            
            round += 1
            logger.info(f"Starting round {round}. Now sending data of shape {pm.shape} to approach.")
            a.receive_predictions_of_new_ensemble_member(pm)

            # define expected number of entries for estimates
            #expected_num_estimates_for_cond = np.minimum(position_wise_thresholds, np.array([round, round**2, round**2, round**2, round**3, round**3, round**4]))
            #print(expected_num_estimates_for_cond)
            #assert np.allclose(expected_num_estimates_for_cond, a.epa.cov_updater_for_conditional_case.num_used_samples_per_cov_estimate)
            
            
            # if this was the first update, retriever the cov updaters
            if round == 1:
                updaters = [
                    a.epa.cov_updater_for_conditional_case,
                    a.epa.cov_updater_for_iid_case_equal_instances,
                    a.epa.cov_updater_for_iid_case_arbitrary_instances
                ]
                last_count_of_observations = [u.num_used_samples_per_cov_estimate for u in updaters]
                last_active_masks = [u.mask_of_active_params.copy() for u in updaters]
            
            else:
                
                for i, (updater, last_count, last_active_mask) in enumerate(zip(updaters, last_count_of_observations, last_active_masks)):
                    logger.info(f"Updater {i}. {last_active_mask} {last_count} {updater.get_highest_order_of_member_combinations_required()}")
                    
                    # check that the updater didn't add samples for any covariance builder that previously was declared inactive.
                    cur_counts = updater.num_used_samples_per_cov_estimate.copy()
                    counts_changes = cur_counts - last_count
                    self.assertTrue(np.all(~last_active_mask | (counts_changes > 0)), f"Observed no change in active parameter. {last_active_mask} (problem in index {np.where(last_active_mask | (counts_changes > 0))[0]}). Counts stayed at {last_count[last_active_mask  | (counts_changes > 0)]}")
                    last_active_mask[:] = updater.mask_of_active_params.copy()
                    last_count[:] = cur_counts
                if round > 20 or a.epa.cov_updater_for_conditional_case.get_highest_order_of_member_combinations_required() == 0:
                    break
        
        # check that all updaters are disabled
        self.assertTrue(updaters[0].is_active)
        self.assertTrue(updaters[0].get_highest_order_of_member_combinations_required() == 2)
        self.assertTrue(updaters[1].is_active)
        self.assertTrue(updaters[1].get_highest_order_of_member_combinations_required() == 1)
        self.assertTrue(updaters[2].is_active)
        self.assertTrue(updaters[2].get_highest_order_of_member_combinations_required() == 1)

    def test_that_cov_updaters_disable_params_upon_saturation(self):

        lim = 50

        a = DatabaseWiseApproach(
            population_mode="stream",
            random_state=0,
            threshold_for_number_of_samples_to_exclude_param=lim,
            logger=approach_logger
        )

        pi = ProblemInstanceWrapperForTesting(
            ensemble_seed=0,
            num_possible_ensemble_members=8,
            validation_size=5
        )
        a.reset()
        a.tell_ground_truth_labels(pi.pi.y_oh_val)
        round = 0

        updaters = None
        last_count_of_observations = None

        for pm in pi.pi.get_prediction_matrix_generator(ensemble_sequence_seed=0, only_validation_data=True):
            round += 1
            logger.info(f"Round {round}. Shape of data is {pm.shape}")
            a.receive_predictions_of_new_ensemble_member(pm)

            for num_update, (name, updater) in enumerate(zip(
                ["conditional", "idd equal", "iid different"],
                [a.epa.cov_updater_for_conditional_case, a.epa.cov_updater_for_iid_case_equal_instances, a.epa.cov_updater_for_iid_case_arbitrary_instances]
            )):
                logger.info(f"Number of items used for estimates in '{name}' cov estimator: {updater.num_used_samples_per_cov_estimate}")

            if round > 60 or a.epa.cov_updater_for_conditional_case.get_highest_order_of_member_combinations_required() == 0:
                break
        
        # check that all updaters have the number of estimates expected
        updaters = [a.epa.cov_updater_for_conditional_case, a.epa.cov_updater_for_iid_case_equal_instances, a.epa.cov_updater_for_iid_case_arbitrary_instances]
        for num_update, (name, updater) in enumerate(zip(
            ["conditional", "idd equal", "iid different"],
            updaters
        )):
            for i, num_estimates in enumerate(updater.num_used_samples_per_cov_estimate):
                if num_update == 2 and i in [3, 4, 6]:
                    continue
                assert num_estimates >= lim, f"Not all cov estimates for '{name}' have at least {lim} entries: Number of estimates for index {i} are only {num_estimates}/{lim}"

        # check that all updaters are disabled
        for updater in updaters:
            self.assertEqual(0, updater.get_highest_order_of_member_combinations_required())
            self.assertFalse(updater.is_active)

    
    def test_correct_behavior_of_ensemble_performance_estimator(self):

        epa = EnsemblePerformanceAssessor(threshold_for_number_of_samples_to_exclude_param=10**8, population_mode="stream")
        
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
        a = self.get_run_approach(param, threshold_for_number_of_samples_to_exclude_param=10**8)

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
        self.assertEqual((7, ), predicted_covs.shape)

        # determine ground truth covariances
        gtt = self.gtc.get_conditional_ground_truth_table()
        true_covariances = self.gtc.get_covariance_terms_for_each_instance_pair(gtt, ddof=0).drop(columns=["i_1", "i_2"]).mean()

        # compare predictions with ground truth in the n x n x 7 covariance terms
        for c, (act_cov, pred_cov) in enumerate(zip(true_covariances, predicted_covs), start=1):
            self.assertAlmostEqual(act_cov, pred_cov, msg=f"Wrong covariance estimate for case {c}.")
        
        # compare predicted (and stored) variance with true variance, and make sure that result is the same as in vectorized one
        v_pred_array = a.estimate_performance_var_in_conditional_setup(t=self.t_checkpoints)
        for t, v_pred_from_array in zip(self.t_checkpoints, v_pred_array):
            v_act = self.gtc.get_true_parameter(param=param, t=t)
            v_pred = a.estimate_performance_var_in_conditional_setup(t=np.array([t]))[0]
            self.assertAlmostEqual(v_act, v_pred, msg=f"Final variance prediciton for V[Z_2,{t}|D_val] is not correct, but the covariance estimates are. So this is a matter of aggregation.")
            self.assertEqual(v_pred_from_array, v_pred)

    
    def test_approximation_quality_for_variance_in_iid_setting(self):
        param = "V[Z_nt]"

        # create a more difficult case here
        self.setUp(n_samples=100, num_different_ensemble_members=5)

        # get ground truth covariances through same approach but without an effective limit
        a_true = self.get_run_approach(param, threshold_for_number_of_samples_to_exclude_param=10**8)
        true_covs_as_array = a_true.xi_covs_in_iid_setting

        # run approximating approach
        allowed_instances = np.array([10**3, 10**3, 10**3, 10**3, 10**4, 10**4, 10**5])
        a_approx = self.get_run_approach(param, threshold_for_number_of_samples_to_exclude_param=allowed_instances)

        # get extracted covs from approach
        predicted_covs = a_approx.xi_covs_in_iid_setting

        # compare predictions with ground truth in the 14 covariance terms
        for i in range(2):
            for c, (predicted_cov, true_cov, tol) in enumerate(zip(
                predicted_covs.reshape(2, -1)[i],
                true_covs_as_array.reshape(2, -1)[i],
                [1, 1, 1, 1, 2, 2, 3]
            )):
                self.assertAlmostEqual(true_cov, predicted_cov, places=tol, msg=f"Wrong estimate for case {c} in {'identical instance' if i == 0 else 'arbitrary instance'} scenario.")
        
        # compare predicted (and stored) variance with true variance (on iid samples from the validation data as population)
        n_checkpoints = np.array([2, 3])
        v_pred_array = a_approx.estimate_performance_var_in_iid_setup(n=n_checkpoints, t=self.t_checkpoints)
        for n, var_predictions_for_n in zip(n_checkpoints, v_pred_array):
            for t, pred_var in zip(self.t_checkpoints, var_predictions_for_n):
                true_var = a_true.estimate_performance_var_in_iid_setup(n=n, t=t)[0, 0]
                if n * t <= 10:
                    places = 1
                elif n * t <= 20:
                    places = 2
                elif n * t <= 100:
                    places = 4
                else:
                    places = 6
                self.assertAlmostEqual(true_var, pred_var, places=places, msg=f"Final variance prediciton for V[Z_{n},{t}] is not precise enough using {allowed_instances} samples.")

    def test_estimate_velocity(self):

        self.setUp(n_samples=64, num_different_ensemble_members=64)
        
        approach = DatabaseWiseApproach(
            estimated_parameters="V[Z_nt]",
            population_mode="stream",
            random_state=0,
            threshold_for_number_of_samples_to_exclude_param=10**7,
            logger=approach_logger
        )

        round = 0
        approach.reset()
        approach.tell_ground_truth_labels(self.y_oh)
        max_degrees_arbitrary = []
        max_degrees_equal = []
        runtimes = []
        print(f"Starting run with a total of {len(self.matrices)} matrices.")
        for pm in range(12):
            round += 1
            print(f"Starting round #{round}")
            t_start = time.time()
            if approach.epa.cov_updater_for_iid_case_arbitrary_instances is not None:
                max_degrees_arbitrary.append(approach.epa.cov_updater_for_iid_case_arbitrary_instances.get_highest_order_of_member_combinations_required())
            if approach.epa.cov_updater_for_iid_case_arbitrary_instances is not None:
                max_degrees_equal.append(approach.epa.cov_updater_for_iid_case_equal_instances.get_highest_order_of_member_combinations_required())
            approach.receive_predictions_of_new_ensemble_member(pm)
            t_end = time.time()
            runtime =  t_end - t_start
            print(f"Finished round #{round} after {runtime}s")
            runtimes.append(runtime)
        
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(runtimes)
        ax.plot(max_degrees_arbitrary)
        ax.plot(max_degrees_equal)
        plt.show()