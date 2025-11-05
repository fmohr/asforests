import numpy as np

#from time import time

import logging

import os
import psutil

from tqdm import tqdm

from experiments.problem_instance.problem_instance import ProblemInstance
from experiments.benchmark.result_storage import ResultStorage

from time import time

class Benchmark:

    def __init__(self,
                 problem_instance: ProblemInstance,
                 captured_parameter,
                 ensemble_sequence_seed=0,
                 ensemble_prefix=None,
                 estimate_checkpoints=None,
                 precision=7,
                 upper_bound_for_sample_size_in_ground_truth_computation=10**8,
                 track_used_resources=False,
                 hooks=[]
                 ):
        
        # configuration variables
        self._problem_instance = problem_instance
        self._ensemble_sequence_seed = ensemble_sequence_seed
        self._ensemble_prefix = ensemble_prefix
        self._prediction_matrix_generator = None
        self.hooks = hooks
        
        self.logger = logging.getLogger("benchmark")
        self.captured_parameter = captured_parameter
        self.upper_bound_for_sample_size_in_ground_truth_computation = upper_bound_for_sample_size_in_ground_truth_computation
        self._estimate_checkpoints = estimate_checkpoints
        self._precision = precision
        self.track_used_resources = track_used_resources

        # state variables
        self._approaches = None
        self._budget = None
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
        return self._budget

    @property
    def result_storage(self):
        return self._result_storage
    
    def reset(self, approaches: dict, ensemble_sequence_seed: int = None):

        # create/reset prediction matrix generator
        if ensemble_sequence_seed is None:
            ensemble_sequence_seed = self._ensemble_sequence_seed
        self._prediction_matrix_id_generator = self.problem_instance.get_prediction_matrix_id_generator(ensemble_sequence_seed)

        # register approaches
        self._approaches = approaches
        for approach in self._approaches.values():
            approach.reset()
            approach.tell_ground_truth_labels(self.problem_instance.y_oh_val)
        
        # we use the database-based approach to estimate the ground truth (only possible if we show exactly once the predictions of all ensemble members, cf unit tests)
        self.logger.info(f"Starting computation of ground truth. Maximum sample size at each stage is {self.upper_bound_for_sample_size_in_ground_truth_computation}")
        pi = self.problem_instance
        _true_parameter_routines = {
            "E[Z_nt]": "means_iid",
            "E[Z_nt|D_val]": "means_cond",
            "V[Z_nt]": "vars_iid",
            "V[Z_nt|D_val]": "vars_cond"
        }
        self._true_parameter = getattr(pi, _true_parameter_routines[self.captured_parameter])
        self.logger.info(f"Ground truth parameter value is: %s", self._true_parameter)

        # reset storage
        self._budget = 0
        self._history_of_member_ids = []
        self._result_storage = ResultStorage(
            true_param_values={self.captured_parameter: self._true_parameter},
            n_checkpoints=self.problem_instance.n_checkpoints,
            t_checkpoints=self.problem_instance.t_checkpoints
            )
    
    def step(self):

        if self._approaches is None:
            raise ValueError("No approaches registered. Use `reset` to define the approaches.")

        self.logger.info(f"Starting round {self._budget + 1}.")

        # update knowledge of all approaches
        member_id = next(self.ensemble_member_id_generator) if (self._ensemble_prefix is None or self._budget >= len(self._ensemble_prefix)) else self._ensemble_prefix[self._budget]
        self._history_of_member_ids.append(member_id)
        matrix = self.problem_instance.predictions_val[member_id]
        self._budget += 1
        if self.track_used_resources:
            self.logger.debug(
                f"Current memory consumption is {self.process.memory_info().rss / (1024 ** 2):.2f}MB. "
                f"Current CPU usage is {self.process.cpu_percent(interval=1.0)}."
            )
        if np.any(np.isnan(matrix)):
            raise ValueError(f"Prediction matrix in round {self._budget} has nan entries.")
        
        for approach_name, approach_obj in self._approaches.items():
            self.logger.debug(f"Stepping {approach_name}.")

            # tell the approach about the new matrix
            approach_obj.receive_predictions_of_new_ensemble_member(matrix)
            
            p = self.captured_parameter

            # ask about the updated opinion on the relevant parameters
            for t in self.problem_instance.t_checkpoints:
                if p == "V[Z_nt]":
                    for n in self.problem_instance.n_checkpoints:
                        t_0 = time()
                        e = approach_obj.estimate_performance_var_in_iid_setup(t=t, n=n)
                        runtime = time() - t_0
                    
                        # add result to storage
                        self._result_storage.add_result(
                            approach=approach_name,
                            budget=self._budget,
                            param=self.captured_parameter,
                            n=n,
                            t=t,
                            estimate=e[0],
                            runtime=runtime
                        )
                else:
                    t_0 = time()
                    if self.captured_parameter == "E[Z_nt]":
                        e = approach_obj.estimate_performance_mean_in_iid_setup(t=t)
                    elif self.captured_parameter == "E[Z_nt|D_val]":
                        e = approach_obj.estimate_performance_mean_in_conditional_setup(t=t)
                    elif self.captured_parameter == "V[Z_nt|D_val]":
                        e = approach_obj.estimate_performance_var_in_conditional_setup(t=t)
                    runtime = time() - t_0
                    
                    # add result to storage
                    self._result_storage.add_result(
                        approach=approach_name,
                        budget=self._budget,
                        param=self.captured_parameter,
                        n=None,
                        t=t,
                        estimate=e[0],
                        runtime=runtime
                    )

        self.logger.info(f"Finished round {self._budget}.")
        for hook in self.hooks:
            hook(self._approaches, self._result_storage)
