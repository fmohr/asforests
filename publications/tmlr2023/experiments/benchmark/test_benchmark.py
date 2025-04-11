import numpy as np
import pandas as pd 
import itertools as it
from tqdm import tqdm

from experiments.benchmark.benchmark import Benchmark, ResultStorage
from experiments.benchmark.approaches import *
from sklearn.datasets import make_classification

from ._util import get_all_ensemble_data_combinations, get_all_ensemble_combinations_on_deviations

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



class TestBenchmark(TestCase):

    def get_standard_benchmark(self, n_samples=10, **kwargs):
        kwargs_default = {
            "data_seed": 0,
            "ensemble_seed": 0,
            "ensemble_sequence_seed": 0,
            "training_instances_per_class": 2,
            "validation_size": 4,
            "num_possible_ensemble_members": 2,
            "max_ground_truth_table_size": 10**4
        }
        num_classes = 2
        kwargs_default.update(kwargs)

        X, y = make_classification(n_classes=num_classes, n_samples=n_samples, n_features=20, random_state=2)

        return Benchmark(
            X=X,
            y=y,
            is_classification=True,
            **kwargs_default
        )

    def test_ability_on_non_standard_data(self):
        openmlid = 188  # eucalyptus
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 50
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True,
            max_ground_truth_table_size=10**3
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({}, t_checkpoints=t_checkpoints)
    
    def test_that_ground_truth_values_are_insensitive_to_change_in_ensemble_sequence_seed(self):
        true_parameters = []
        for ensemble_seed in range(2):
            b = Benchmark(
                openmlid=61,
                data_seed=0,
                ensemble_seed=0,
                ensemble_sequence_seed=ensemble_seed,
                num_possible_ensemble_members=5,
                training_instances_per_class=10,
                validation_size=10,
                is_classification=True,
                max_ground_truth_table_size=10**3
            )
            
            # run benchmark twice for 10 iterations (10 ensemble members)
            b.reset({}, t_checkpoints=[10])

            true_parameters.append(b._true_parameters)
        
        for k in true_parameters[0].keys():
            self.assertEqual(true_parameters[0][k], true_parameters[1][k], f"Ground truth value of {k} changes with the ensemble sequence seed.")

    def test_that_ground_truth_values_are_sensitive_to_change_in_ensemble_seed(self):
        true_parameters = []
        for ensemble_seed in range(2):
            b = Benchmark(
                openmlid=61,
                data_seed=0,
                ensemble_seed=ensemble_seed,
                ensemble_sequence_seed=0,
                num_possible_ensemble_members=5,
                training_instances_per_class=10,
                validation_size=10,
                is_classification=True
            )
            
            b.reset(approaches={}, t_checkpoints=[5])

            true_parameters.append(b._true_parameters)
        
        for k in true_parameters[0].keys():
            self.assertNotEqual(true_parameters[0][k], true_parameters[1][k], f"Ground truth value of {k} is invariant to the ensemble seed.")

    def test_that_ground_truth_values_are_sensitive_to_change_in_data_seed(self):
        true_parameters = []
        for data_seed in range(2):
            b = Benchmark(
                openmlid=61,
                data_seed=data_seed,
                ensemble_seed=0,
                ensemble_sequence_seed=0,
                num_possible_ensemble_members=5,
                training_instances_per_class=10,
                validation_size=10,
                is_classification=True
            )
            
            b.reset(approaches={}, t_checkpoints=[5])

            true_parameters.append(b._true_parameters)
        
        for k in true_parameters[0].keys():
            self.assertNotEqual(true_parameters[0][k], true_parameters[1][k], f"Ground truth value of {k} is invariant to the data seed.")
    
    def test_correctness_of_ground_truth_conditional(self):
        
        t_domain = np.arange(2, 7)

        b = self.get_standard_benchmark()
        
        ddof = 0
        for index_t, t in enumerate(t_domain):
            b.reset(approaches={}, t_checkpoints=list(t_domain))

            # get ground truth
            df_worlds = get_all_ensemble_combinations_on_deviations(b._deviations[:, b._indices_val], t=t)
            true_mean = df_worlds["z"].mean()
            true_var = df_worlds["z"].var(ddof=ddof)

            # get prediction of benchmark
            mean_according_to_benchmark = b._true_parameters["E[Z_nt|D_val]"][index_t]
            var_according_to_benchmark = b._true_parameters["V[Z_nt|D_val]"][index_t]  # no index_n given here since n is determine by |D_val|
            self.assertAlmostEqual(true_mean, mean_according_to_benchmark)
            self.assertAlmostEqual(true_var, var_according_to_benchmark)

    def test_correctness_of_ground_truth_iid(self):
        
        b = self.get_standard_benchmark(max_ground_truth_table_size=10**6)
        t_domain = np.arange(2, 6)
        b.reset(approaches={}, t_checkpoints=list(t_domain))

        ddof = 0
        for index_t, t in enumerate(t_domain):

            # get ground truth
            df_worlds = get_all_ensemble_data_combinations(b._deviations, n=2, t=t)
            true_mean = df_worlds["z"].mean()
            true_var = df_worlds["z"].var(ddof=ddof)

            # get prediction of benchmark
            mean_according_to_benchmark = b._true_parameters["E[Z_nt]"][index_t]
            var_according_to_benchmark = b._true_parameters["V[Z_nt]"][index_t]
            self.assertAlmostEqual(true_mean, mean_according_to_benchmark, msg=f"E[Z_nt] is {true_mean} but was estimated with {mean_according_to_benchmark}.")
            self.assertAlmostEqual(true_var, var_according_to_benchmark, msg=f"V[Z_nt] is {true_var} but was estimated with {var_according_to_benchmark}.")

    def test_ground_truth_approximation_for_big_datasets(self):

        t = np.array([10])
        n_samples = 800

        # get ground truth
        b = self.get_standard_benchmark(n_samples=n_samples, max_ground_truth_table_size=10**8)
        b.reset(approaches={}, t_checkpoints=t)
        true_mean, true_var = b.get_true_performance_mean_on_iid_data(t=t), b.get_true_performance_var_for_two_instances_on_iid_data(t=t)

        # approximate true parameters
        b = self.get_standard_benchmark(n_samples=n_samples, max_ground_truth_table_size=10**6)
        b.reset(approaches={}, t_checkpoints=t)
        approximated_mean, approximated_var = b.get_true_performance_mean_on_iid_data(t=t), b.get_true_performance_var_for_two_instances_on_iid_data(t=t)
        
        for i, _t in enumerate(t):
            self.assertAlmostEqual(true_mean[i], approximated_mean[i])
            self.assertAlmostEqual(true_var[i], approximated_var[i])

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(random_state=0, num_resamples=100)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
    ])
    def test_approach_functionality_in_conditional_setting(self, a_name, a_obj):
        
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**2
        for _ in tqdm(range(num_steps)):
            b.step()
        
        # check that reasonable estimates have been given
        num_classes = b.y_oh.shape[1]
        estimates = a_obj.estimate_performance_mean_in_conditional_setup(t_checkpoints)
        for i, e in enumerate(estimates):
            self.assertTrue(e > 0)
            self.assertTrue(e < num_classes, f"Estimates should be below the number of classes {num_classes} but {a_name} estimated: {estimates}")
            if i > 0:
                self.assertTrue(e < estimates[i-1], f"Estimates should be monotonically decreasing but {a_name} estimated: {estimates}")  # check monotonicity

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(random_state=0, num_resamples=100)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
    ])
    def test_approach_functionality_in_iid_setting(self, a_name, a_obj):
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**2
        for _ in tqdm(range(num_steps)):
            b.step()
        
        # check that reasonable estimates have been given
        estimates = a_obj.estimate_performance_mean_in_iid_setup(t_checkpoints)
        num_classes = b.y_oh.shape[1]
        for i, e in enumerate(estimates):
            self.assertTrue(e > 0)
            self.assertTrue(e < num_classes, f"Estimates should be below the number of classes {num_classes} but {a_name} estimated: {estimates}")
            if i > 0:
                self.assertTrue(e < estimates[i-1], f"Estimates should be monotonically decreasing but {a_name} estimated: {estimates}")  # check monotonicity

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(num_resamples=1)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
    ])

    def test_result_extraction(self, a_name, a_obj):
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**2
        for _ in tqdm(range(num_steps)):
            b.step()
        
        # extract results
        for t in t_checkpoints:
            df = b.result_storage.get_estimates_from_approach_for_checkpoint(a_name, t)
            self.assertEqual(num_steps, len(df))

            df = b.result_storage.get_errors_from_approach_for_checkpoint(a_name, t)
            self.assertEqual(num_steps, len(df))

    def test_reproducibility(self):
        
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(random_state=0, num_resamples=1),
            "theorem with datasets": DatabaseWiseApproach(random_state=0, upper_bound_for_sample_size=10**10),
            "parametric model": ParametricModelApproach(random_state=0, num_simulated_ensembles=100)
        }

        # run benchmark twice for 10 iterations (10 ensemble members)
        storages = []
        for _ in range(2):
            b.reset(approaches, t_checkpoints=t_checkpoints)
            for _ in tqdm(range(10**1)):
                b.step()
            storages.append(b.result_storage)
        
        # check equality of storages
        def assertDictEqualRecursive(dict1, dict2, prefix=""):
            """Recursively assert that two dictionaries are identical."""
            self.assertEqual(set(dict1.keys()), set(dict2.keys()), "Keys mismatch")
            
            for key in dict1:
                value1, value2 = dict1[key], dict2[key]
                
                if isinstance(value1, dict) and isinstance(value2, dict):
                    assertDictEqualRecursive(value1, value2, prefix=prefix + f"/{key}")  # Recursive check
                else:
                    self.assertEqual(value1, value2, f"Mismatch at key '{prefix}/{key}'")
        assertDictEqualRecursive(storages[0]._estimates, storages[1]._estimates)
    
    def test_serialization_and_deserialization_of_results(self):
        
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(num_resamples=1),
            "theorem with datasets": DatabaseWiseApproach(upper_bound_for_sample_size=10**10),
            "parametric model": ParametricModelApproach(num_simulated_ensembles=100)
        }

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset(approaches, t_checkpoints=t_checkpoints)
        for _ in tqdm(range(10**1)):
            b.step()
        
        # test that the unserialized serialized result storage has the same state as the fresh result storage.
        storage = b.result_storage
        recovered_storage = ResultStorage.unserialize(storage.serialize())
        for i, (v1, v2) in enumerate(zip(storage.t_checkpoints, recovered_storage.t_checkpoints)):
            self.assertEqual(v1, v2)
        for k in storage.true_param_values:
            self.assertTrue(k in recovered_storage.true_param_values)
            for t, v1, v2 in zip(storage.t_checkpoints, storage.true_param_values[k], recovered_storage.true_param_values[k]):
                self.assertEqual(v1, v2)
        for i, (v1, v2) in enumerate(zip(storage.approach_names, recovered_storage.approach_names)):
            self.assertEqual(v1, v2)
        self.assertDictEqual(storage._estimates, recovered_storage._estimates)
        self.assertDictEqual(storage._runtimes, recovered_storage._runtimes)
    
    def test_merge_result_storages(self):
        
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(num_resamples=1),
            "theorem with datasets": DatabaseWiseApproach(upper_bound_for_sample_size=10**10),
            "parametric model": ParametricModelApproach(num_simulated_ensembles=100)
        }

        # run benchmark in isolation for each approach
        result_storages = {}
        budgets = set()
        for a in approaches:
            b.reset({a: approaches[a]}, t_checkpoints=t_checkpoints)
            for _ in tqdm(range(10**1)):
                b.step()
            assert len(b.result_storage.approach_names) == 1 and b.result_storage.approach_names[0] == a
            result_storages[a] = b.result_storage
            budgets |= b.result_storage.budgets
        
        # merge the result storages
        rs_merged = ResultStorage.merge(result_storages.values())

        # now check that the estimates in the merged store are available and identical
        approach_names = sorted(approaches.keys())
        self.assertEqual(str(approach_names), str(rs_merged.approach_names))
        self.assertEqual(len(budgets), len(rs_merged.budgets))
        for a in approaches:
            for p in rs_merged.true_param_values:
                for v1, v2 in zip(rs_merged.true_param_values[p], result_storages[a].true_param_values[p]):
                    self.assertEqual(v1, v2)
            for v1, v2 in zip(rs_merged.t_checkpoints, result_storages[a].t_checkpoints):
                self.assertEqual(v1, v2)
            self.assertDictEqual(rs_merged._estimates[a], result_storages[a]._estimates[a])
            self.assertDictEqual(rs_merged._runtimes[a], result_storages[a]._runtimes[a])

    def test_rename_approach(self):
        b = self.get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        approaches = {
            "bootstrapping": BootstrappingApproach(num_resamples=1),
            "theorem with datasets": DatabaseWiseApproach(upper_bound_for_sample_size=10**10),
            "parametric model": ParametricModelApproach(num_simulated_ensembles=100)
        }

        # run benchmark
        b.reset(approaches, t_checkpoints=t_checkpoints)
        for _ in tqdm(range(10**1)):
            b.step()
        
        # check that we can properly rename an approach
        rs = b.result_storage
        n_from = "bootstrapping"
        n_to = "bootstrapping1"
        rs.rename_approach(n_from, n_to)
        self.assertTrue(n_to in rs.approach_names)
        self.assertTrue(n_to in rs._estimates)
        self.assertTrue(n_to in rs._runtimes)
        self.assertFalse(n_from in rs._estimates)
        self.assertFalse(n_from in rs._runtimes)
        for n in [a for a in approaches if a != n_from]:
            self.assertTrue(n in rs.approach_names)
            self.assertTrue(n in rs._estimates)
            self.assertTrue(n in rs._runtimes)

    @parameterized.expand([(a, b, c) for (a, b), c in it.product([
        #("bootstrapping", BootstrappingApproach(random_state=0)),
        ("theorem with datasets", DatabaseWiseApproach(create_estimates_for_iid_scenario=False, upper_bound_for_sample_size=10**10)),
        #("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
        #("parametric diff model", ParametricDifferenceModelApproach(random_state=0))
    ], [
        "E[Z_nt|D_val]",
        #"V[Z_nt|D_val]"
        ])])
    def test_that_approach_converges_to_no_error_on_validation_data(self, a_name, a_obj, param):
        """
        All approaches should converge to an estimation error of 0 for both E[Z_nt|D_val] and V[Z_nt|D_val] when conditioning on concrete (known) data

        Args:
            a_name (_type_): _description_
            a_obj (_type_): _description_
        """

        factor_to_increase_checkpoints = 5
        required_factor_of_improvement = 2
        num_macro_steps = 6

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]
        e_checkpoints = [factor_to_increase_checkpoints**i for i in range(num_macro_steps + 1)]

        histories = []
        seeds = [0, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for seed in seeds:
            b = self.get_standard_benchmark(
                ensemble_sequence_seed=seed,
                captured_parameters=[param],
                estimate_checkpoints=e_checkpoints
            )

            # run benchmark twice for 10 iterations (10 ensemble members)
            b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
            
            # prepare ground truth and state variables
            target = b._true_parameters[param]
            last_estimation_error = np.inf
            logger.info(f"Testing whether {a_name} converges against true values {target} of {param} on seed {seed}")
            history = []
            for e_checkpoint in e_checkpoints:

                # configure the granularity of the approach for this stage
                if isinstance(a_obj, BootstrappingApproach):
                    a_obj.num_resamples = int(np.sqrt(e_checkpoint) // 2)
                    a_obj.bootstrap_size = int(4 * a_obj.num_resamples)
                if isinstance(a_obj, ParametricDifferenceModelApproach):
                    a_obj.num_simulated_ensembles = e_checkpoint
                
                # advance the approaches until next checkpoint
                while b._t < e_checkpoint:
                    b.step()
                
                logger.info(f"Getting estimates at checkpoint.")
                #estimation = a_obj.estimate_performance_var_in_conditional_setup(t_checkpoints)
                estimation = np.array([b._result_storage.get_estimates_from_approach_for_checkpoint(a_name, t=_t)[param].iloc[-1] for _t in t_checkpoints])
                max_estimation_error = np.max(np.abs(target - estimation))
                logger.info(f"Estimate with {b._t} ensemble members: {estimation}. Highest error: {max_estimation_error}")
                history.append(max_estimation_error)
            histories.append(history)
            
            # update mean history (but no check yet until we have run all the seeds)
            mean_history = np.array(histories).mean(axis=0)
            logger.info(f"Average error history is {np.round(mean_history, 5)}")
            actual_average_improvement_rates = []
            for i, mean_max_estimation_error_in_step in enumerate(mean_history):
                if i > 1:  # skip first round since this could be very good just by guessing or accidently initializing with a good value
                    actual_average_improvement_rates.append(float(last_estimation_error / mean_max_estimation_error_in_step))
                last_estimation_error = mean_max_estimation_error_in_step
            logger.info(f"Average improvement rates over {len(histories)} seeds are {np.round(actual_average_improvement_rates, 2)}")
        
        # check that average improvement is by the required factor
        self.assertLessEqual(required_factor_of_improvement, min(actual_average_improvement_rates))


    """
    I believe this test is obsolte since we already have checks for ground truth correspondence.

    def test_correct_expected_values_of_generated_ensembles(self):

        t = 5

        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        num_possible_ensemble_members = 2
        training_instances_per_class = 10
        validation_size = 5
        n_estimators = 10**4

        estimates = []
        ground_truth = None

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=None,
            num_possible_ensemble_members=num_possible_ensemble_members,
            validation_size=validation_size,
            training_instances_per_class=training_instances_per_class,
            is_classification=True
        )

        import matplotlib.pyplot as plt

        for ensemble_sequence_seed in tqdm(range(n_estimators)):    

            # get generator for the estimates of the approach on the given problem
            t_checkpoints = [t]

            approaches = {"dummy": DummyApproach()} # we use this approach here since it just takes the empirical mean

            # reset benchmark and memorize ground truth (only in first run)
            b.reset(approaches, t_checkpoints=t_checkpoints, ensemble_sequence_seed=ensemble_sequence_seed)
            if ground_truth is None:
                ground_truth = b._true_parameters["E[Z_nt|D_val]"]

            # train dummy
            for _ in range(t):
                b.step()
            estimates.append(b.result_storage.get_estimates_from_approach_for_checkpoint(approach_name="dummy", t=t)["E[Z_nt|D_val]"].values[-1])

            if len(estimates) > 1 and ensemble_sequence_seed % 100 == 0:
                print(len(estimates), np.abs(ground_truth - np.mean(estimates)))
                for e in range(1, 5):
                    if ensemble_sequence_seed >= 10**(e + 1):
                        self.assertTrue(np.isclose(ground_truth, np.mean(estimates), atol=10**-e))
        """