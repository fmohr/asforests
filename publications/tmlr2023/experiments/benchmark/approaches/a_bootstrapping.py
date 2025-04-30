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

        # get ensembles with their member predictions as a 5D tensor
        ensemble_descriptors_through_indices = self.random_state.randint(0, b, size=(self.num_resamples, self.bootstrap_size, max(t)))
        ensemble_member_predictions = matrices[ensemble_descriptors_through_indices.ravel()].reshape(ensemble_descriptors_through_indices.shape + matrices.shape[1:])
        
        means = []
        variances = []
        for size in t:
            ensemble_predictions = ensemble_member_predictions[:, :, :size, :, :].mean(axis=2)
            ensemble_errors = ((ensemble_predictions - self.y_oh)**2).mean(axis=2).sum(axis=2)
            means.append(ensemble_errors.mean(axis=1).mean())
            variances.append(ensemble_errors.var(axis=1).mean())
        self._means = np.array(means)
        self._vars = np.array(variances)

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
