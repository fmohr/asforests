import numpy as np
import pandas as pd 
import itertools as it
from tqdm import tqdm

from experiments.benchmark.benchmark import Benchmark, ResultStorage
from experiments.benchmark.approaches import *
from sklearn.datasets import make_classification

from _ground_truth_computer import GroundTruthComputer

from unittest import TestCase
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
bm_logger.setLevel(logging.WARNING)

# configure logger for tester
approach_logger = logging.getLogger("approach")
approach_logger.handlers.clear()
approach_logger.addHandler(ch)
approach_logger.setLevel(logging.WARNING)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.INFO)


def get_standard_benchmark(n_samples=10, **kwargs):
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


class TestBenchmark(TestCase):
    
    @parameterized.expand([(a, b, c) for (a, b), c in it.product([
        ("bootstrapping", BootstrappingApproach(random_state=0, num_resamples=10, bootstrap_size=10)),
        #("theorem with datasets", DatabaseWiseApproach(create_estimates_for_iid_scenario=False, upper_bound_for_sample_size=10**10)),
        #("parametric model", ParametricDifferenceModelApproach(num_simulated_ensembles=8))
        #("parametric diff model", ParametricDifferenceModelApproach(random_state=0, logger=approach_logger))
    ], [
        "E[Z_nt|D_val]",
        "V[Z_nt|D_val]"
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
        logger.info(f"{e_checkpoints=}")

        histories = []
        seeds = [0, 3, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
        for seed in seeds:
            b = get_standard_benchmark(
                #openmlid=54,
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
                logger.info(f"Stepping the approaches until size {e_checkpoint}")
                while b._t < e_checkpoint:
                    b.step()
                    logger.debug(f"Step {b._t} ready.")
                
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
            logger.info(f"Average improvement rates over {len(histories)} seeds are {np.round(actual_average_improvement_rates, 2)} (required is {required_factor_of_improvement})")
        
        # check that average improvement is by the required factor
        self.assertLessEqual(required_factor_of_improvement, min(actual_average_improvement_rates))