from experiments.benchmark.tests.util import ApproachTestClass

from experiments.benchmark.benchmark import Benchmark
from experiments.benchmark.tests.util import ProblemInstanceWrapperForTesting
from experiments.benchmark.approaches.a_parametric_diff import ParametricDifferenceModelApproach

import logging
import numpy as np

from tqdm import tqdm

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
approach_logger = logging.getLogger("tested_approach.parametric_approach")
approach_logger.handlers.clear()
approach_logger.addHandler(ch)
approach_logger.setLevel(logging.WARNING)

class TestParametricApproach(ApproachTestClass):

    def get_approach(self, seed, estimated_parameters):
        return ParametricDifferenceModelApproach(
            estimated_parameters=estimated_parameters,
            random_state=seed,
            num_simulated_ensembles=10**3,
            logger=approach_logger
        )

    def adjust_approach_object_for_evaluation_on_conditional_convergence_test_checkpoint(self, approach, e_checkpoint):
        approach.num_simulated_ensembles = e_checkpoint
    

    def test_that_higher_number_of_simulations_improves_result(self):

        wrapper = ProblemInstanceWrapperForTesting(ensemble_seed=0)
        pi = wrapper.pi
        n_checkpoints = wrapper.n_checkpoints
        t_checkpoints = wrapper.t_checkpoints
        gen = pi.get_prediction_matrix_generator(only_validation_data=True)

        # check performances across different repetitions
        est_errs_mean_iid = []
        est_errs_var_iid = []
        est_errs_mean_cond = []
        est_errs_var_cond = []
        for _ in tqdm(range(100)):
            pm_sequence = [next(gen) for _ in range(1000)]

            # create approaches with different numbers of simulations
            budgets = [2, 10**3]
            approaches = [
                ParametricDifferenceModelApproach(
                    num_simulated_ensembles=num_sims,
                    random_state=0
                )
                for num_sims in budgets
            ]

            # reset all of them
            for approach in approaches:
                approach.reset()
                approach.tell_ground_truth_labels(pi.y_oh_val)  # generator only uses validation data, so we here should also only tell validation labels

            # advance all of them
            for pm in pm_sequence:
                for a_obj in approaches:
                    a_obj.receive_predictions_of_new_ensemble_member(pm)
            
            # get ground truth
            means_iid = pi.means_iid
            means_cond = pi.means_cond
            vars_iid = pi.vars_iid
            vars_cond = pi.vars_cond
            
            # compare ground truth to estimates
            est_errs_mean_iid_for_seed = []
            est_errs_var_iid_for_seed = []
            est_errs_mean_cond_for_seed = []
            est_errs_var_cond_for_seed =  []
            for approach in approaches:
                est_errs_mean_iid_for_seed.append(approach.estimate_performance_mean_in_iid_setup(t=t_checkpoints) - means_iid)
                est_errs_var_iid_for_seed.append(approach.estimate_performance_var_in_iid_setup(t=t_checkpoints, n=n_checkpoints) - vars_iid)
                est_errs_mean_cond_for_seed.append(approach.estimate_performance_mean_in_conditional_setup(t=t_checkpoints) - means_cond)
                est_errs_var_cond_for_seed.append(approach.estimate_performance_var_in_conditional_setup(t=t_checkpoints) - vars_cond)
            est_errs_mean_iid.append(est_errs_mean_iid_for_seed)
            est_errs_var_iid.append(est_errs_var_iid_for_seed)
            est_errs_mean_cond.append(est_errs_mean_cond_for_seed)
            est_errs_var_cond.append(est_errs_var_cond_for_seed)

        def verify_improvement(errors_for_t):
            prev_val = None
            for b, v in zip(budgets, errors_for_t):
                if prev_val is not None:
                    assert prev_val > v, f"Error for budget {b} must be strictly lower than error with lower budget but is not"
                prev_val = v
        
        # check that error on E[Z_nt] decreases
        est_errs_mean_iid = np.abs(np.array(est_errs_mean_iid)) # shape (#repetitions, #budgets, #t)
        avg_error_for_mean_iid_by_t = est_errs_mean_iid.mean(axis=0).T # shape (#t, #budgets)
        for t, errors_for_t in zip(t_checkpoints, avg_error_for_mean_iid_by_t):
            verify_improvement(errors_for_t)

        # check that error on V[Z_nt] decreases
        est_errs_var_iid = np.abs(np.array(est_errs_var_iid)) # shape (#repetitions, #budgets, #n, #t)
        avg_error_for_var_iid_by_n_and_t = est_errs_var_iid.mean(axis=0).transpose(1, 2, 0) # shape (#n, #t, #budgets)
        for n, errors_for_n in zip(n_checkpoints, avg_error_for_var_iid_by_n_and_t):
            for t, errors_for_t in zip(t_checkpoints, errors_for_n):
                verify_improvement(errors_for_t)

        # check that error on E[Z_nt|cond] decreases
        est_errs_mean_cond = np.abs(np.array(est_errs_mean_cond)) # shape (#repetitions, #budgets, #t)
        avg_error_for_mean_cond_by_t = est_errs_mean_cond.mean(axis=0).T # shape (#t, #budgets)
        for t, errors_for_t in zip(t_checkpoints, avg_error_for_mean_cond_by_t):
            verify_improvement(errors_for_t)

        # check that error on V[Z_nt|cond] decreases
        est_errs_var_cond = np.abs(np.array(est_errs_var_cond)) # shape (#repetitions, #budgets, #t)
        avg_error_for_var_cond_by_t = est_errs_var_cond.mean(axis=0).T # shape (#t, #budgets)
        for t, errors_for_t in zip(t_checkpoints, avg_error_for_var_cond_by_t):
            verify_improvement(errors_for_t)