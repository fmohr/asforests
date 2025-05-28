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