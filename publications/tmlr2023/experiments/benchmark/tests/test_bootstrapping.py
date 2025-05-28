from approaches import DatabaseWiseApproach, BootstrappingApproach
from tqdm import tqdm
import numpy as np

from asforests.cb_computer import EnsemblePerformanceAssessor

from _ground_truth_computer import GroundTruthComputer

from sklearn.datasets import make_classification
from sklearn.model_selection import StratifiedShuffleSplit
from experiments.benchmark._util import get_unique_prediction_matrices

from unittest import TestCase
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

from experiments.benchmark.tests.util import get_problem_instance_for_openmlid, get_standard_benchmark, ProblemInstanceWrapperForTesting, ApproachTestClass
from experiments.benchmark.approaches.a_bootstrapping import BootstrappingApproach


class TestBootstrappingApproach(ApproachTestClass):

    def get_approach(self, seed, estimated_parameters):
        return BootstrappingApproach(
            estimated_parameters=estimated_parameters,
            bootstrap_size=10**1,
            num_resamples=10**1,
            random_state=seed
        )

    def adjust_approach_object_for_evaluation_on_conditional_convergence_test_checkpoint(self, approach, e_checkpoint):
        approach.num_resamples = int(np.sqrt(e_checkpoint) // 2)
        approach.bootstrap_size = int(4 * approach.num_resamples)