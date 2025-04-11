import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .approach import Approach
from scipy.optimize import OptimizeWarning
import warnings

import logging
from tqdm import tqdm


class ParametricDifferenceModelApproach(Approach):

    def __init__(
            self,
            num_simulated_ensembles=1,
            with_replacement=False,
            anchors="power",
            show_progress=False,
            **kwargs
            ):
        super().__init__(**kwargs)

        # config
        self.anchors = anchors
        self.num_simulated_ensembles = num_simulated_ensembles
        self.with_replacement = with_replacement
        self.show_progress = show_progress

        # state variables
        self.updated_estimates = None
    
    def reset(self):

        # state
        super().reset()
        self.prediction_matrices = []
        self.updated_estimates = {}

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        self.prediction_matrices.append(prediction_matrix)
        self.updated_estimates = {}
    
    def _check_param_coverage(self, param):
        if param not in self.estimated_parameters:
            self.logger.warning(
                f"Parameter {param} not in configured estimated params {self.estimated_parameters}. "
                "This is not a problem for this approach, but this indicates some ill configuration."
            )

    def estimate_performance_mean_in_iid_setup(self, t):
        param = "E[Z_nt]"
        self._check_param_coverage(param)
        
        if param not in self.updated_estimates:
            
            # if we do not have enough observations, return 0
            # TODO: return empirical mean
            b = len(self.prediction_matrices)
            if b < 2:
                self.updated_estimates[param] = np.zeros(len(t))
                return self.updated_estimates[param]
            
            # create permutations
            ensembles = [self.random_state.choice(range(b), size=b, replace=self.with_replacement) for _ in range(self.num_simulated_ensembles)]

            # compute data for parametric learning problem
            sizes = []
            errors = []
            self.logger.info(f"Computing database")
            for ensemble in ensembles:

                if self.anchors == "full":
                    ensemble_prediction_matrix = np.zeros(self.prediction_matrices[0].shape)
                    for s, i in enumerate(ensemble, start=1):
                        ensemble_prediction_matrix += (self.prediction_matrices[i] - ensemble_prediction_matrix) / s
                        error_of_this_ensemble_per_target = ((ensemble_prediction_matrix - self.y_oh)**2).mean(axis=0)
                        sizes.append(s)
                        errors.append(error_of_this_ensemble_per_target)
                
                elif self.anchors.startswith("power"):
                    matrices = np.array(self.prediction_matrices)
                    schedule = sorted(set([int(np.round(2**(i / 2))) for i in range(int(2 * np.log2(b) + 1))]))
                    for size in schedule:
                        ensemble_prediction_matrix = matrices[ensemble[:size]].mean(axis=0)
                        error_of_this_ensemble_per_target = ((ensemble_prediction_matrix - self.y_oh)**2).mean(axis=0)
                        sizes.append(size)
                        errors.append(error_of_this_ensemble_per_target)
                
            errors = np.array(errors).T
            self.logger.info(f"Done. Database has {len(sizes)} entries.")

            # Define parametric function
            def model(t, a, b):
                return a + b / t

            # estimate parameters
            self.logger.info(f"Now fitting {len(errors)} models, one per target.")
            estimates = np.zeros(len(t))
            for j, target_errors in enumerate(errors):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=OptimizeWarning)
                    (a, b), covariance = curve_fit(model, sizes, target_errors)
                estimates += a + b / t
            self.logger.info(f"Done, updating parameter estimates to {estimates}")
            self.updated_estimates[param] = estimates
        return self.updated_estimates[param]
    
    def estimate_performance_mean_in_conditional_setup(self, t):
        param = "E[Z_nt|D_val]"
        self._check_param_coverage(param)

        if param not in self.updated_estimates:

            """
                In this case, we build a model for each instance/target combination

                This is because we know that E[Z_nt|D_val] can be decomposed into E[Z_ijt], where i is a specific instance and j the target.
            """

            # estimate dummy if we do not have enough observations
            budget = len(self.prediction_matrices)
            if budget < 2:
                self.updated_estimates[param] = np.ones(len(t))
                return self.updated_estimates[param]

            # Define parametric function
            def model(t, a, b):
                return a + b / t
            
            checkpoints_for_budget = [10**i for i in range(int(np.log10(budget)) + 1)]
            if budget not in checkpoints_for_budget:
                checkpoints_for_budget.append(budget)

            # estimate parameters
            pbar = tqdm(total=self.y_oh.shape[0] * self.y_oh.shape[1], disable=not self.show_progress)
            estimates = np.zeros(len(t))
            matrices = np.array(self.prediction_matrices)
            for i in range(self.y_oh.shape[0]):
                for j in range(self.y_oh.shape[1]):
                    ensemble_member_predictions_on_instance_for_label = matrices[:, i, j]
                    ground_truth_on_instance_for_label = self.y_oh[i, j]

                    # compute for different ensembles of different sizes the left hand side
                    self.logger.info(f"Computing errors of ensembles at different sizes.")
                    sizes = []
                    errors = []
                    for e_id in tqdm(range(self.num_simulated_ensembles), disable=not self.show_progress):
                        ensemble = self.random_state.choice(range(budget), size=budget, replace=self.with_replacement)
                        for s in checkpoints_for_budget:
                            #ensemble_prediction_matrix += (self.prediction_matrices[ensemble[s - 1]] - ensemble_prediction_matrix) / s
                            ensemble_prediction_matrix = ensemble_member_predictions_on_instance_for_label[ensemble[:s]].mean(axis=0)
                            errors.append((ensemble_prediction_matrix - ground_truth_on_instance_for_label)**2)
                            sizes.append(s)
                    
                    dataset = pd.DataFrame({"t": sizes, "y": errors})
                    if dataset["y"].min() == dataset["y"].max():
                        a, b = dataset["y"].min(), 0
                    else:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", category=OptimizeWarning)
                            (a, b), _ = curve_fit(model, dataset["t"], dataset["y"])
                    estimates += a + b / t
                    pbar.update(1)
            pbar.close()
            self.updated_estimates[param] = estimates / self.y_oh.shape[0] # divide by n
            self.logger.info(f"Estimates updated to {self.updated_estimates}")
        return self.updated_estimates[param]

    def estimate_performance_var_for_two_instances_in_iid_setup(self, t):
        param = "V[Z_nt]"
        self._check_param_coverage(param)

        if param not in self.updated_estimates:
            
            self.updated_estimates[param] = np.ones(len(t))
        return self.updated_estimates[param]
    
    def estimate_performance_var_in_conditional_setup(self, t):
        param = "V[Z_nt|D_val]"
        self._check_param_coverage(param)

        if param not in self.updated_estimates:
            
            self.updated_estimates[param] = np.ones(len(t))
        return self.updated_estimates[param]
        
