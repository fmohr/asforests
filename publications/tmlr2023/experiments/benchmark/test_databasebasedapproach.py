from approaches import DatabaseWiseApproach
from benchmark import Benchmark
from test_benchmark import get_standard_benchmark
from tqdm import tqdm
import numpy as np
from _ground_truth_computer import GroundTruthComputer

from unittest import TestCase
from parameterized import parameterized

import itertools as it

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


class TestDatabaseBasedApproach(TestCase):

    """
        This test generates a special sequence of ensemble members, namely each of them exactly once.
        Thereby, the approach has seen all ensemble members exactly once and should, by coincidence, estimate the exactly correct variance
    """
    def test_correct_estimation_of_variance_in_conditional_setup(self):

        param = "V[Z_nt|D_val]"

        a = DatabaseWiseApproach(
            estimated_parameters=[param],
            population_mode="stream",
            single_data_point_per_ensemble_member=False,
            create_estimates_for_iid_scenario=True
        )

        num_possible_ensemble_members = 5

        b = get_standard_benchmark(
            ensemble_seed=2,
            num_possible_ensemble_members=num_possible_ensemble_members,
            ensemble_prefix=list(range(num_possible_ensemble_members)),
            max_ground_truth_table_size=10**6
        )
        t_checkpoints = np.array([1, 10, 100])
        b.reset(approaches={"my": a}, t_checkpoints=t_checkpoints)

        for _ in tqdm(range(num_possible_ensemble_members)):
            b.step()

        # get extracted covs from approach
        predicted_covs = a.xi_covs_in_conditional_setting

        # determine ground truth covariances
        gtc = GroundTruthComputer(deviations=b._deviations[:, b._indices_val])
        gtt = gtc.get_conditional_ground_truth_table()
        gt_covariances = gtc.get_covariance_terms_for_each_instance_pair(gtt, ddof=0)
        
        # compare predictions with ground truth in the n x n x 7 covariance terms
        for (i1, i2), df_covariances in gt_covariances.groupby(["i_1", "i_2"]):
            i1 -= 1
            i2 -= 1
            self.assertEqual(1, len(df_covariances))
            covs = df_covariances.drop(columns=["i_1", "i_2"]).iloc[0].values
            for c, cov in enumerate(covs):
                self.assertAlmostEqual(cov, predicted_covs[i1, i2, c], msg=f"Wrong estimate for case {c} on instance pair {i1}/{i2}")
        
        # compare predicted (and stored) variance with true variance
        for t, v_act in zip(t_checkpoints, b._true_parameters[param]):
            v_pred = a.estimate_performance_var_in_conditional_setup(t=np.array([t]))[0]
            self.assertAlmostEqual(v_act, v_pred, msg=f"Final variance prediciton for V[Z_2,{t}|D_val] is not correct, but the covariance estimates are. So this is a matter of aggregation.")

            v_pred_stored = b.result_storage.get_estimates_from_approach_for_checkpoint(approach_name="my", t=t)[param].iloc[-1]
            self.assertAlmostEqual(v_act, v_pred_stored, msg=f"Final variance prediciton for V[Z_2,{t}|D_val] is not correct, but the covariance estimates are. So this is a matter of aggregation.")
            
    def test_correct_estimation_of_variances_in_iid_setup(self):

        param = "V[Z_nt]"
        
        a = DatabaseWiseApproach(
            estimated_parameters=[param],
            population_mode="stream",
            single_data_point_per_ensemble_member=False,
            create_estimates_for_iid_scenario=True
        )

        num_possible_ensemble_members = 5

        b = get_standard_benchmark(
            ensemble_seed=2,
            num_possible_ensemble_members=num_possible_ensemble_members,
            ensemble_prefix=list(range(num_possible_ensemble_members)),
            max_ground_truth_table_size=10**6
        )
        t_checkpoints = np.arange(1, 6)
        b.reset(approaches={"my": a}, t_checkpoints=t_checkpoints)

        for _ in tqdm(range(num_possible_ensemble_members)):
            b.step()

        # get extracted covs from approach
        predicted_covs = a.xi_covs_in_iid_setting

        # determine ground truth covariances (iid from the validation data; THIS IS DIFFERENT FROM THE BENCHMARK, WHICH USES ALL DATA POINTS)
        gtc = GroundTruthComputer(deviations=b._deviations[:, b._indices_val])
        gtt = gtc.get_ground_truth_table_under_sample_iid_assumption()
        gt_covariances = gtc.get_covariance_terms_for_each_instance_pair(gtt, ddof=0)
        true_covs_as_array = gt_covariances[gt_covariances["i_1"] == 1].drop(columns=["i_1", "i_2"]).values.flatten()

        # compare predictions with ground truth in the 14 covariance terms
        for c, cov in enumerate(true_covs_as_array):
            self.assertAlmostEqual(cov, predicted_covs[c], msg=f"Wrong estimate for case {c}")
        
        # compare predicted (and stored) variance with true variance (on iid samples from the validation data as population)
        for t in t_checkpoints:
            true_var = gtc.get_all_ensemble_data_combinations(t=t, n=2, compute_deviations=False)["z"].var(ddof=0)
            pred_var = a.estimate_performance_var_for_two_instances_in_iid_setup(t=np.array([t]))[0]
            self.assertAlmostEqual(true_var, pred_var, msg=f"Final variance prediciton for V[Z_2,{t}] is not correct, but the covariance estimates are. So this is a matter of aggregation.")

            v_pred_stored = b.result_storage.get_estimates_from_approach_for_checkpoint(approach_name="my", t=t)[param].iloc[-1]
            self.assertAlmostEqual(true_var, v_pred_stored, msg=f"Final variance prediciton for V[Z_2,{t}] is not correct, but the covariance estimates are. So this is a matter of aggregation.")