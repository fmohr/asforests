from approaches import BootstrappingApproach
import numpy as np

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