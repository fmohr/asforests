import numpy as np
from tqdm import tqdm
import json

from pathlib import Path

from experiments.benchmark.benchmark import Benchmark, ResultStorage
from experiments.benchmark.approaches import *
from experiments.benchmark.tests.util import get_problem_instance_for_openmlid, get_standard_benchmark, ProblemInstanceWrapperForTesting

from unittest import TestCase
import pytest
from parameterized import parameterized

import logging


# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for benchmark
bm_logger = logging.getLogger("benchmark")
bm_logger.handlers.clear()
bm_logger.addHandler(ch)
bm_logger.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.INFO)


class TestBenchmark(TestCase):

    def test_ability_on_non_standard_data(self):
        openmlid = 188  # eucalyptus
        n_checkpoints=[2]
        t_checkpoints=[10, 100, 1000]
        
        file, pi = get_problem_instance_for_openmlid(
            openmlid=openmlid,
            n_checkpoints=n_checkpoints,
            t_checkpoints=t_checkpoints
        )

        # create 
        b = Benchmark(
            problem_instance=pi,
            ensemble_sequence_seed=0
        )

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset({})

        # store problem instance with ground truth
        if not file.exists():
            with open(file, "w") as f:
                json.dump(pi.to_dict(), f)
    
    
    def test_ability_to_create_necessary_number_of_distinct_ensemble_members(self):
        
        # get problem instance
        w = ProblemInstanceWrapperForTesting(ensemble_seed=0)
        pi = w.pi

        # check whether benchmark can be reset and whether we can extract ground truth values
        b = Benchmark(problem_instance=pi)
        b.reset({})
        assert np.array_equal(pi.means_iid, b._true_parameters["E[Z_nt]"])
        assert np.array_equal(pi.means_cond, b._true_parameters["E[Z_nt|D_val]"])
        assert np.array_equal(pi.vars_iid, b._true_parameters["V[Z_nt]"])
        assert np.array_equal(pi.vars_cond, b._true_parameters["V[Z_nt|D_val]"])

    def test_result_extraction(self):

        # get benchmark
        w = ProblemInstanceWrapperForTesting(ensemble_seed=0)
        pi = w.pi
        b = Benchmark(problem_instance=pi)

        # run benchmark twice for 10 iterations (10 ensemble members)
        approaches = {
            "bootstrapping": BootstrappingApproach(random_state=0, bootstrap_size=10, num_resamples=1),
            "parametric model": ParametricDifferenceModelApproach(random_state=0, num_simulated_ensembles=100)
        }
        b.reset(approaches)
        num_steps = 10**1
        for _ in tqdm(range(num_steps)):
            b.step()
        
        # extract results
        for a_name in approaches.keys():
            for n in pi.n_checkpoints:
                for t in pi.t_checkpoints:
                    df = b.result_storage.get_results_from_approach_for_checkpoint(approach_name=a_name, n_for_var_in_iid_case=n, t=t)
                    self.assertEqual(num_steps * len(b.captured_parameters), len(df))

                    df = b.result_storage.get_errors_from_approach_for_checkpoint(approach_name=a_name, n_for_var_in_iid_case=n, t=t)
                    self.assertEqual(num_steps * len(b.captured_parameters), len(df))

    def test_reproducibility(self):
        
        b = get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        n_checkpoints=[2]
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(random_state=0, bootstrap_size=10, num_resamples=1),
            "parametric model": ParametricDifferenceModelApproach(random_state=0, num_simulated_ensembles=100)
        }

        # run benchmark twice for 10 iterations (10 ensemble members)
        storages = []
        for _ in range(2):
            b.reset(approaches)
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
        self.assertTrue(np.array_equal(storages[0].results["estimate"], storages[1].results["estimate"]))
    
    def test_serialization_and_deserialization_of_results(self):
        
        b = get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        n_checkpoints=[2]
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(bootstrap_size=1, num_resamples=1),
            "parametric model": ParametricDifferenceModelApproach(num_simulated_ensembles=100)
        }

        # run benchmark twice for 10 iterations (10 ensemble members)
        b.reset(approaches)
        for _ in tqdm(range(10**1)):
            b.step()
        
        # test that the unserialized serialized result storage has the same state as the fresh result storage.
        storage = b.result_storage
        recovered_storage = ResultStorage.unserialize(storage.serialize())
        for i, (v1, v2) in enumerate(zip(storage.t_checkpoints, recovered_storage.t_checkpoints)):
            self.assertEqual(v1, v2)
        for p in storage.true_param_values:
            self.assertTrue(p in recovered_storage.true_param_values)
            if p == "V[Z_nt]":
                for i_n, n in enumerate(n_checkpoints):
                    for i_t, t in enumerate(t_checkpoints):
                        self.assertEqual(storage.true_param_values[p][i_n, i_t], recovered_storage.true_param_values[p][i_n, i_t])
            else:
                for t, v1, v2 in zip(storage.t_checkpoints, storage.true_param_values[p], recovered_storage.true_param_values[p]):
                    self.assertEqual(v1, v2)
        for i, (v1, v2) in enumerate(zip(storage.approach_names, recovered_storage.approach_names)):
            self.assertEqual(v1, v2)
        self.assertEqual(len(storage.results), len(recovered_storage.results))
        self.assertEqual(list(storage.results.index), list(recovered_storage.results.index))
        self.assertEqual(list(storage.results.columns), list(recovered_storage.results.columns))
        for row1, row2 in zip(storage.results.values, recovered_storage.results.values):
            self.assertEqual(list(row1), list(row2))
        self.assertTrue(storage.results.equals(recovered_storage.results))
    
    def test_merge_result_storages(self):
        
        b = get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        n_checkpoints = [2]
        t_checkpoints = [10, 100, 1000]
        approaches = {
            "bootstrapping": BootstrappingApproach(bootstrap_size=10, num_resamples=1),
            "parametric model": ParametricDifferenceModelApproach(num_simulated_ensembles=100)
        }

        # run benchmark in isolation for each approach
        result_storages = {}
        budgets = set()
        for a in approaches:
            b.reset({a: approaches[a]})
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
        self.assertGreater(len(budgets), 0)
        self.assertEqual(len(budgets), len(rs_merged.budgets))
        for a in approaches:
            for p in rs_merged.true_param_values:
                for v1, v2 in zip(rs_merged.true_param_values[p], result_storages[a].true_param_values[p]):
                    assert (v1, v2)
            for v1, v2 in zip(rs_merged.t_checkpoints, result_storages[a].t_checkpoints):
                self.assertEqual(v1, v2)
            df_approach_in_merged = rs_merged.results[rs_merged.results["approach"] == a]
            df_approach_in_isolation = result_storages[a].results
            self.assertEqual(len(df_approach_in_isolation), len(df_approach_in_merged))
            self.assertTrue(np.array_equal(df_approach_in_isolation.values, df_approach_in_merged.values))

    def test_rename_approach(self):
        b = get_standard_benchmark()

        # get generator for the estimates of the approach on the given problem
        n_checkpoints = [2]
        t_checkpoints = [10, 100, 1000]

        approaches = {
            "bootstrapping": BootstrappingApproach(bootstrap_size=10, num_resamples=1),
            "parametric model": ParametricDifferenceModelApproach(num_simulated_ensembles=100)
        }

        # run benchmark
        b.reset(approaches)
        for _ in tqdm(range(10**1)):
            b.step()
        
        # check that we can properly rename an approach
        rs = b.result_storage
        n_from = "bootstrapping"
        n_to = "bootstrapping1"

        num_entries_before = np.count_nonzero(rs.results["approach"] == n_from)

        rs.rename_approach(n_from, n_to)
        self.assertTrue(n_to in rs.approach_names)
        self.assertTrue(n_from not in rs.approach_names)
        
        num_entries_after = np.count_nonzero(rs.results["approach"] == n_to)

        self.assertEqual(num_entries_before, num_entries_after)
