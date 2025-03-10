import numpy as np
from tqdm import tqdm

from experiments.benchmark.benchmark import Benchmark
from experiments.benchmark.approaches import *

from unittest import TestCase


class TestBenchmark(TestCase):

    def test_reproducibility(self):
        openmlid = 61
        data_seed = 0
        ensemble_seed = 0
        validation_size = 20
        application_size = 50

        b = Benchmark(
            openmlid=openmlid,
            data_seed=data_seed,
            ensemble_seed=ensemble_seed,
            validation_size=validation_size,
            application_size=application_size,
            is_classification=True
        )

        # get generator for the estimates of the approach on the given problem
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(num_resamples=1),
            "theorem with datasets": DatabaseWiseApproach(upper_bound_for_sample_size=10**10),
            "parametric model": ParametricModelApproach(rs=np.random.RandomState(0), num_simulated_ensembles=100)
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