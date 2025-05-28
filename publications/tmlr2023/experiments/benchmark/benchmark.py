import numpy as np

from time import time

import logging

import os
import psutil

from tqdm import tqdm

from experiments.problem_instance.problem_instance import ProblemInstance
from experiments.benchmark.result_storage import ResultStorage
from experiments.benchmark.approaches.a_fromdatabase import DatabaseWiseApproach # used to compute ground truths as this is much more efficient than the naive way


class Benchmark:

    def __init__(self,
                 problem_instance: ProblemInstance,
                 ensemble_sequence_seed=0,
                 ensemble_prefix=None,
                 captured_parameters=["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"],
                 estimate_checkpoints=None,
                 precision=7,
                 upper_bound_for_sample_size_in_ground_truth_computation=10**8,
                 track_used_resources=False
                 ):
        
        # configuration variables
        self._problem_instance = problem_instance
        self._ensemble_sequence_seed = ensemble_sequence_seed
        self._ensemble_prefix = ensemble_prefix
        self._prediction_matrix_generator = None
        
        self.logger = logging.getLogger("benchmark")
        self.captured_parameters = captured_parameters
        self.upper_bound_for_sample_size_in_ground_truth_computation = upper_bound_for_sample_size_in_ground_truth_computation
        self._estimate_checkpoints = estimate_checkpoints
        self._precision = precision
        self.track_used_resources = track_used_resources

        # state variables
        self._approaches = None
        self._t_checkpoints = None
        self._t = None
        self._result_storage = None
        self.process = psutil.Process(os.getpid()) # get process for memory surveillance

    @property
    def problem_instance(self) -> ProblemInstance:
        return self._problem_instance

    @property
    def ensemble_member_id_generator(self):
        return self._prediction_matrix_id_generator
    
    @property
    def t(self):
        return self._t

    @property
    def result_storage(self):
        return self._result_storage
    
    def reset(self, approaches: dict, n_checkpoints: list, t_checkpoints: list, ensemble_sequence_seed: int = None):

        # create/reset prediction matrix generator
        if ensemble_sequence_seed is None:
            ensemble_sequence_seed = self._ensemble_sequence_seed
        self._prediction_matrix_id_generator = self.problem_instance.get_prediction_matrix_id_generator(ensemble_sequence_seed)

        # register approaches
        self._approaches = approaches
        for approach in self._approaches.values():
            approach.reset()
            approach.tell_ground_truth_labels(self.problem_instance.y_oh_val)
        
        # register check points and compute true values for those checkpoints
        if isinstance(t_checkpoints, int):
            t_checkpoints = np.array([t_checkpoints])
        if isinstance(t_checkpoints, list):
            t_checkpoints = np.array(t_checkpoints)
        if not isinstance(t_checkpoints, np.ndarray) or not t_checkpoints.dtype == int:
            raise ValueError(f"t_checkpoints must be an integer, a list of integers, or a np array of type int but is {type(t_checkpoints)}")
        self._t_checkpoints = t_checkpoints

        # we use the database-based approach to estimate the ground truth (only possible if we show exactly once the predictions of all ensemble members, cf unit tests)
        self.logger.info(f"Starting computation of ground truth. Maximum sample size at each stage is {self.upper_bound_for_sample_size_in_ground_truth_computation}")
        pi = self.problem_instance
        self._true_parameters = {
            "E[Z_nt]": pi.means_iid,
            "E[Z_nt|D_val]": pi.means_cond,
            "V[Z_nt]": pi.vars_iid,
            "V[Z_nt|D_val]": pi.vars_cond
        }
        self.logger.info(f"Ground truth parameter values are: {self._true_parameters}")

        # reset storage
        self._t = 0
        self._history_of_member_ids = []
        self._result_storage = ResultStorage(
            true_param_values=self._true_parameters,
            n_checkpoints=n_checkpoints,
            t_checkpoints=t_checkpoints
            )
    
    def step(self):

        if self._approaches is None:
            raise ValueError("No approaches registered. Use `reset` to define the approaches.")

        self.logger.info(f"Starting round {self._t}.")

        # update knowledge of all approaches
        member_id = next(self.ensemble_member_id_generator) if (self._ensemble_prefix is None or self._t >= len(self._ensemble_prefix)) else self._ensemble_prefix[self._t]
        self._history_of_member_ids.append(member_id)
        matrix = self.problem_instance.predictions_val[member_id]
        self._t += 1
        if self.track_used_resources:
            self.logger.debug(
                f"Current memory consumption is {self.process.memory_info().rss / (1024 ** 2):.2f}MB. "
                f"Current CPU usage is {self.process.cpu_percent(interval=1.0)}."
            )
        
        if np.any(np.isnan(matrix)):
            raise ValueError(f"Prediction matrix in round {self._t} has nan entries.")

        do_update_estimates = self._estimate_checkpoints is None or self._t in self._estimate_checkpoints

        for approach_name, approach_obj in self._approaches.items():
            self.logger.debug(f"Stepping {approach_name}.")
            keys_and_methods_available = {
                "add": (lambda: approach_obj.receive_predictions_of_new_ensemble_member(matrix), False),
                #"update_iid": (approach_obj._update_estimates_for_iid, False),
                #"update_cond": (approach_obj._update_estimates_for_conditional, False),
                "E[Z_nt|D_val]": (approach_obj.estimate_performance_mean_in_conditional_setup, True),
                "V[Z_nt|D_val]": (approach_obj.estimate_performance_var_in_conditional_setup, True),
                "E[Z_nt]": (approach_obj.estimate_performance_mean_in_iid_setup, True),
                "V[Z_nt]": (approach_obj.estimate_performance_var_in_iid_setup, True)
            }

            enabled_keys = ["add"] + [p for p in ["E[Z_nt|D_val]", "E[Z_nt]", "V[Z_nt|D_val]", "V[Z_nt]"] if p in approach_obj.estimated_parameters and p in self.captured_parameters]
            keys_and_methods_applied = {k: keys_and_methods_available[k] for k in enabled_keys}

            if do_update_estimates:
                self.logger.info(f"Requesting estimates for {list(keys_and_methods_applied.keys())} from {approach_name}")
            for p, (m, has_estimate) in keys_and_methods_applied.items():
                t0 = time()
                if has_estimate and do_update_estimates:
                    self.logger.debug(f"Requesting estimates for {p} from {approach_name}")
                    if p == "V[Z_nt]":
                        e = m(self.problem_instance.n_checkpoints, self.problem_instance.t_checkpoints)
                        assert (len(self.problem_instance.n_checkpoints), len(self.problem_instance.t_checkpoints)) == e.shape, f"Wrong shape returned by approach {approach_name} for V[Z_nt]. Should be {(len(self.problem_instance.n_checkpoints), len(self.problem_instance.t_checkpoints))} but was {e.shape}"
                    else:
                        e = m(self.problem_instance.t_checkpoints)
                        assert (len(self.problem_instance.t_checkpoints), ) == e.shape, f"Wrong shape returned by approach {approach_name} for {p}. Should be {(len(self.problem_instance.t_checkpoints), )} but was {e.shape}"
                    self.logger.debug(f"{approach_name} estimates {e} for {p}")
                elif not has_estimate:
                    e = m()
                t1 = time()
                if has_estimate and do_update_estimates:
                    assert isinstance(e, np.ndarray), f"Returned estimates must be a numpy array, but {approach_name} returned {type(e)} for {p}"
                    for j, t in enumerate(self.problem_instance.t_checkpoints):    
                        if p == "V[Z_nt]":
                            for i, n in enumerate(self.problem_instance.n_checkpoints):
                                self._result_storage.add_result(approach=approach_name, param=p, budget=self._t, n=n, t=t, estimate=np.round(e[i, j], self._precision), runtime=np.round(t1 - t0, self._precision))
                        else:
                            self._result_storage.add_result(approach=approach_name, param=p, budget=self._t, n=None, t=t, estimate=np.round(e[j], self._precision), runtime=np.round(t1 - t0, self._precision))

            #if do_update_estimates:
                #print(e)
                #self.logger.info(f"Storing estimates {estimates} for approach {approach_name} with runtimes {runtimes}")
                #self._result_storage.add_result(
                #    approach_name=approach_name,
                #    n=None,
                #    t=self.t,
                #    estimates,
                #    runtimes
                #)
            #self.logger.debug(f"Stepped {approach_name}. Runtimes: {runtimes}. Estimates are {estimates}")
        self.logger.info(f"Step finished {self._t} finished.")
