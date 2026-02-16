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

# configure logger for tester
approach_logger = logging.getLogger("tested_approach.bootstrapping")
approach_logger.handlers.clear()
approach_logger.addHandler(ch)
approach_logger.setLevel(logging.DEBUG)

from experiments.benchmark.tests.util import ApproachTestClass
from experiments.benchmark.approaches.a_bootstrapping import BootstrappingApproach


class TestBootstrappingApproach(ApproachTestClass):

    def get_approach(self, seed, estimated_parameters):
        return BootstrappingApproach(
            estimated_parameters=estimated_parameters,
            bootstrap_size=10**2, # use bootstrap size of 100 just to be sure
            num_resamples=10**2, # use 100 samples, just to be sure
            random_state=seed,
            logger=approach_logger
        )

    def adjust_approach_object_for_evaluation_on_conditional_convergence_test_checkpoint(self, approach, e_checkpoint):
        approach.num_resamples = np.maximum(1, int(np.sqrt(e_checkpoint) // 2))
        approach.bootstrap_size = int(4 * approach.num_resamples)