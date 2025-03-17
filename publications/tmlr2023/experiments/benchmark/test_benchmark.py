import numpy as np
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
        training_size = 50
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            training_size=training_size,
            validation_size=validation_size,
            is_classification=True
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({}, t_checkpoints=t_checkpoints)
        

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(num_resamples=1)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=100))
    ])
    def test_approach_functionality(self, a_name, a_obj):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        training_size = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            training_size=training_size,
            validation_size=validation_size,
            is_classification=True
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**1
        for _ in tqdm(range(num_steps)):
            b.step()

    @parameterized.expand([
        ("bootstrapping", BootstrappingApproach(num_resamples=1)),
        ("theorem with datasets", DatabaseWiseApproach(upper_bound_for_sample_size=10**10)),
        ("parametric model", ParametricModelApproach(num_simulated_ensembles=100))
    ])
    def test_result_extraction(self, a_name, a_obj):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        training_size = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            validation_size=validation_size,
            training_size=training_size,
            is_classification=True
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({a_name: a_obj}, t_checkpoints=t_checkpoints)
        num_steps = 10**1
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
        training_size = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            training_size=training_size,
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
        assertDictEqualRecursive(storages[0].estimates, storages[1].estimates)
    
    def test_serialization_and_deserialization_of_results(self):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        training_size = 10
        validation_size = 20

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            training_size=training_size,
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
        for i, (v1, v2) in enumerate(zip(storage.true_param_values, recovered_storage._true_param_values)):
            self.assertEqual(v1, v2)
        for i, (v1, v2) in enumerate(zip(storage.approach_names, recovered_storage.approach_names)):
            self.assertEqual(v1, v2)
        for i, (v1, v2) in enumerate(zip(storage.t_checkpoints, recovered_storage.t_checkpoints)):
            self.assertEqual(v1, v2)
        self.assertDictEqual(storage._estimates, recovered_storage._estimates)
        self.assertDictEqual(storage._runtimes, recovered_storage._runtimes)
