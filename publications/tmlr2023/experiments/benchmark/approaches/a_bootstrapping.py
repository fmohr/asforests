from .approach import Approach
import numpy as np
from scipy.stats import bootstrap


class BootstrappingApproach(Approach):

    def __init__(self, bootstrap_size, num_resamples, use_caching=True, **kwargs):
        super().__init__(**kwargs)
        self.prediction_matrices = None
        self.bootstrap_size = bootstrap_size
        self.num_resamples = num_resamples
        self.use_caching = use_caching
        
        # state variables
        self._means = self._vars = None
    
    def reset(self):
        super().reset()
        self.prediction_matrices = []

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):

        # add prediction matrix
        self.prediction_matrices.append(prediction_matrix)
        self._means = self._vars = None

    def _check_param_coverage(self, param):
        if param not in self.estimated_parameters:
            self.logger.warning(
                f"Parameter {param} not in configured estimated params {self.estimated_parameters}. "
                "This is not a problem for this approach, but this indicates some ill configuration."
            )

    def _update_estimates(self, t):
        if isinstance(t, int):
            t = [t]

        b = len(self.prediction_matrices)
        if b < 2:
            self._means = self._vars = np.zeros((len(t), ))
            return
        
        matrices = np.array(self.prediction_matrices)
        means = []
        variances = []
        for size in t:
            means_of_bootstrap_samples = []
            variances_of_bootstrap_samples = []
            for _ in range(self.num_resamples):
                ensemble_definitions = self.random_state.choice(range(b), size=(self.bootstrap_size, size), replace=True)
                errors_of_ensembles_on_revealed_data = []
                
                for ensemble_members in ensemble_definitions:
                    mean_prediction = matrices[ensemble_members].mean(axis=0)
                    errors_of_ensembles_on_revealed_data.append(((mean_prediction - self.y_oh)**2).mean(axis=0).sum())
                means_of_bootstrap_samples.append(np.mean(errors_of_ensembles_on_revealed_data))
                variances_of_bootstrap_samples.append(np.var(errors_of_ensembles_on_revealed_data))
            means.append(means_of_bootstrap_samples)
            variances.append(variances_of_bootstrap_samples)
        self._means = np.mean(means, axis=1)
        self._vars = np.mean(variances, axis=1)

    def estimate_performance_mean_in_iid_setup(self, t):
        self._check_param_coverage("E[Z_nt]")
        if self._means is None or not self.use_caching:
            self._update_estimates(t)
        return self._means

    def estimate_performance_mean_in_conditional_setup(self, t):
        self._check_param_coverage("E[Z_nt|D_val]")
        if self._means is None or not self.use_caching:
            self._update_estimates(t)
        return self._means

    def estimate_performance_var_for_two_instances_in_iid_setup(self, t):
        self._check_param_coverage("V[Z_nt]")
        if self._means is None or not self.use_caching:
            self._update_estimates(t)
        return self._vars

    def estimate_performance_var_in_conditional_setup(self, t):
        self._check_param_coverage("V[Z_nt|D_val]")
        if self._vars is None or not self.use_caching:
            self._update_estimates(t)
        return self._vars
