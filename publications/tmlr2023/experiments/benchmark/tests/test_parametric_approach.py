from approaches import DatabaseWiseApproach, BootstrappingApproach
from tqdm import tqdm
import numpy as np

from asforests.cb_computer import EnsemblePerformanceAssessor

from _ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices
from experiments.benchmark.tests.util import ApproachTestClass

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

from experiments.benchmark.benchmark import Benchmark
from experiments.benchmark.tests.util import get_problem_instance_for_openmlid, get_standard_benchmark, ProblemInstanceWrapperForTesting
from experiments.benchmark.approaches.a_parametric_diff import ParametricDifferenceModelApproach

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

# configure logger for tester
approach_logger = logging.getLogger("tested_approach")
approach_logger.handlers.clear()
approach_logger.addHandler(ch)
approach_logger.setLevel(logging.WARNING)

class TestParametricApproach(ApproachTestClass):

    def get_approach(self, seed, estimated_parameters):
        return ParametricDifferenceModelApproach(
            estimated_parameters=estimated_parameters,
            random_state=seed,
            num_simulated_ensembles=10**2,
            logger=approach_logger
        )

    def adjust_approach_object_for_evaluation_on_conditional_convergence_test_checkpoint(self, approach, e_checkpoint):
        approach.num_simulated_ensembles = e_checkpoint
    

    def test_that_higher_number_of_simulations_improves_result(self):

        wrapper = ProblemInstanceWrapperForTesting(ensemble_seed=0)

        benchmark = Benchmark(
            problem_instance=wrapper.pi,
            ensemble_sequence_seed=0
        )

        gen = wrapper.pi.get_prediction_matrix_generator(only_validation_data=True)
        pm_sequence = [next(gen) for _ in range(100)]

        approaches = {
            f"a{num_sims}": ParametricDifferenceModelApproach(
                num_simulated_ensembles=num_sims,
                random_state=0
            )
            for num_sims in [10, 100]
        }

        benchmark.reset(approaches=approaches)

        for _ in range(10):
            benchmark.step()
            print(benchmark.result_storage.get_errors_on_highest_budget(params=["E[Z_nt]"]))