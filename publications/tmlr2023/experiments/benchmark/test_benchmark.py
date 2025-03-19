import numpy as np
import itertools as it
from tqdm import tqdm

from experiments.benchmark.benchmark import Benchmark, ResultStorage
from experiments.benchmark.approaches import *

from unittest import TestCase
from parameterized import parameterized


class TestBenchmark(TestCase):

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
            is_classification=True
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
                is_classification=True
            )
            
            b.reset(approaches={}, t_checkpoints=[10])

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
            
            b.reset(approaches={}, t_checkpoints=[10])

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
            
            b.reset(approaches={}, t_checkpoints=[10])

            true_parameters.append(b._true_parameters)
        
        for k in true_parameters[0].keys():
            self.assertNotEqual(true_parameters[0][k], true_parameters[1][k], f"Ground truth value of {k} is invariant to the data seed.")
    
    def test_correctness_of_ground_truth_conditional(self):
        
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=None,
            num_possible_ensemble_members=10,
            validation_size=validation_size,
            training_instances_per_class=training_instances_per_class,
            is_classification=True
        )
        t_domain = np.arange(1, 21)
        b.reset(approaches={}, t_checkpoints=list(t_domain))

        # test correctness of E[Z_nt|D_val]. Thanks to the condition, the variance become independent
        deviations_on_validation_data = b._deviations[:, b._indices_val]
        tree_deviation_mean = deviations_on_validation_data.mean(axis=0)
        tree_deviation_var = deviations_on_validation_data.var(axis=0)
        ensemble_performance_mean = (tree_deviation_mean**2).mean(axis=0).sum() + tree_deviation_var.mean(axis=0).sum() / t_domain
        for gt, gt_according_to_benchmark in zip(ensemble_performance_mean, b._true_parameters["E[Z_nt|D_val]"]):
            self.assertEqual(gt, gt_according_to_benchmark)

    def test_correctness_of_ground_truth_iid(self):
        
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=None,
            num_possible_ensemble_members=10,
            validation_size=validation_size,
            training_instances_per_class=training_instances_per_class,
            is_classification=True
        )
        t_domain = np.arange(1, 21)
        b.reset(approaches={}, t_checkpoints=list(t_domain))

        # test correctness of E[Z_nt|D_val]. Thanks to the condition, the variance become independent
        tree_deviation_mean = b._deviations.mean(axis=(0, 1))
        tree_deviation_var = b._deviations.var(axis=(0, 1))
        tree_deviation_covs = []
        for j in range(b._deviations.shape[2]):
            tree_deviation_cov_col1 = []
            tree_deviation_cov_col2 = []
            for s1, s2 in it.combinations(range(b._deviations.shape[0]), 2):
                tree_deviation_cov_col1.extend(b._deviations[s1, :, j])
                tree_deviation_cov_col2.extend(b._deviations[s2, :, j])
            tree_deviation_covs.append(np.cov(tree_deviation_cov_col1, tree_deviation_cov_col2, rowvar=False)[0, 1])
        tree_deviation_covs = np.array(tree_deviation_covs)

        ensemble_performance_mean = (tree_deviation_mean**2).sum() + tree_deviation_var.sum() / t_domain + tree_deviation_covs.sum() * (1 - 1 / t_domain)
        for gt, gt_according_to_benchmark in zip(ensemble_performance_mean, b._true_parameters["E[Z_nt]"]):
            self.assertAlmostEqual(gt, gt_according_to_benchmark)

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(random_state=0, num_resamples=100)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
    ])
    def test_approach_functionality_in_conditional_setting(self, a_name, a_obj):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**2
        for _ in tqdm(range(num_steps)):
            b.step()
        
        # check that reasonable estimates have been given
        estimates = a_obj.estimate_performance_mean_in_conditional_setup(t_checkpoints)
        for i, e in enumerate(estimates):
            self.assertTrue(e > 0)
            self.assertTrue(e < 1)
            if i > 0:
                self.assertTrue(e < estimates[i-1])  # check monotonicity

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(random_state=0, num_resamples=100)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
    ])
    def test_approach_functionality_in_iid_setting(self, a_name, a_obj):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**2
        for _ in tqdm(range(num_steps)):
            b.step()
        
        # check that reasonable estimates have been given
        estimates = a_obj.estimate_performance_mean_in_iid_setup(t_checkpoints)
        for i, e in enumerate(estimates):
            self.assertTrue(e > 0)
            self.assertTrue(e < 1)
            if i > 0:
                self.assertTrue(e < estimates[i-1])  # check monotonicity

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(num_resamples=1)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=8))
    ])
    def test_result_extraction(self, a_name, a_obj):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            validation_size=validation_size,
            training_instances_per_class=training_instances_per_class,
            is_classification=True
        )

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
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True
        )

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
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True
        )

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
        
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            training_instances_per_class=training_instances_per_class,
            validation_size=validation_size,
            is_classification=True
        )

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
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        ensemble_sequence_seed = 0
        training_instances_per_class = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            ensemble_sequence_seed=ensemble_sequence_seed,
            num_possible_ensemble_members=5,
            validation_size=validation_size,
            training_instances_per_class=training_instances_per_class,
            is_classification=True
        )

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