from .approach import Approach
import numpy as np
from scipy.stats import bootstrap

from experiments.benchmark.tests.util import get_problem_instance_for_openmlid, get_standard_benchmark


class BootstrappingApproach(Approach):

    def __init__(self, bootstrap_size, num_resamples, sample_in_instance_space=True, use_caching=True, **kwargs):
        super().__init__(**kwargs)
        self.prediction_matrices = None
        self.bootstrap_size = bootstrap_size
        self.num_resamples = num_resamples
        self.sample_in_instance_space = sample_in_instance_space
        self.use_caching = use_caching
        
        # state variables
        self._means_iid = self._vars_iid = None
        self._means_cond = self._vars_cond = None
    
    def reset(self):
        super().reset()
        self.prediction_matrices = []

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):

        # add prediction matrix
        self.prediction_matrices.append(prediction_matrix)
        self._means_iid = self._vars_iid = None
        self._means_cond = self._vars_cond = None

    def _check_param_coverage(self, param):
        if param not in self.estimated_parameters:
            self.logger.warning(
                f"Parameter {param} not in configured estimated params {self.estimated_parameters}. "
                "This is not a problem for this approach, but this indicates some ill configuration."
            )

    def _update_iid_estimates(self, n, t):
        self.logger.info(f"Starting updating estimates with bootstrapping.")
        t = np.asarray(t).reshape(-1)
        if n is None:
            n = self.prediction_matrices[0].shape[0]
        n = np.asarray(n).reshape(-1)

        # initialize mean and var dictionaries if necessary
        if self._means_iid is None:
            self._means_iid =  {} # keys will be values of t
        if self._vars_iid is None:
            self._vars_iid = {} # keys will be pairs (n, t)

        b = len(self.prediction_matrices)
        if b < 2:
            for _t in t:
                self._means_iid[_t] = 0 if b == 0 else (self.prediction_matrices[0]**2).mean(axis=0).sum()
                for _n in n:
                    self._vars_iid[(_n, _t)] = 0
            self.logger.info(f"Finished updating estimates with bootstrapping (since only one ensemble member is known, bootstrapping was skipped).")
            return
        
        matrices = np.array(self.prediction_matrices)

        means = []
        variances = []
        self.logger.debug(f"Now sampling {self.num_resamples} times {self.bootstrap_size} ensembles of size {max(t)}")
        if self.sample_in_instance_space:
            for i in range(self.num_resamples):
                self.logger.debug(f"Creating %s-th bootstrap sample", i)

                # create random ensemble of the maximum given size
                ensemble_descriptors_through_indices = self.random_state.randint(0, b, size=(self.bootstrap_size, max(t)))
                ensemble_member_predictions = matrices[ensemble_descriptors_through_indices.ravel()].reshape(ensemble_descriptors_through_indices.shape + matrices.shape[1:])

                scores = np.zeros((self.bootstrap_size, len(n), len(t)))
                for i_t, size in enumerate(t):
                    for i_e, ensemble_description in enumerate(ensemble_member_predictions):
                        instances_addressed_in_this_ensemble = self.random_state.randint(0, ensemble_description.shape[1], size=max(n))
                        ensemble_predictions_on_selected_instances = ensemble_description[:size, instances_addressed_in_this_ensemble, :].mean(axis=0)
                        for i_n, num_instances in enumerate(n):
                            scores[i_e, i_n, i_t] = (((ensemble_predictions_on_selected_instances[:num_instances] - self.y_oh[instances_addressed_in_this_ensemble][:num_instances])**2).mean(axis=0).sum())
                means.append(scores.mean(axis=0))
                variances.append(scores.var(axis=0))
            
            means_across_bootstrapsamples = np.mean(means, axis=0)
            vars_across_bootstrapsamples = np.mean(variances, axis=0)

            for i_n, _n in enumerate(n):
                for i_t, _t in enumerate(t):
                    if _t not in self._means_iid:
                        self._means_iid[_t] = means_across_bootstrapsamples[i_n, i_t]
                    if (_n, _t) not in self._vars_iid:
                        self._vars_iid[(_n, _t)] = vars_across_bootstrapsamples[i_n, i_t]
        
        # if we do not sample in instance space, we always take exactly the given validation data
        else:
            raise RuntimeError("Not correctly implemented")

            # compute errors for different sub-sizes of this ensemble on the given data
            means_for_round = []
            vars_for_round = []
            for size in t:
                ensemble_predictions = ensemble_member_predictions[:, :size, :, :].mean(axis=1)
                ensemble_errors = ((ensemble_predictions - self.y_oh)**2).mean(axis=1).sum(axis=1)
                means_for_round.append(ensemble_errors.mean())
                vars_for_round.append(ensemble_errors.var())
            means.append(means_for_round)
            variances.append(vars_for_round)        
            self._means_cond = np.array(means).mean(axis=0)
            self._vars_cond = np.array(variances).mean(axis=0)
        self.logger.info(f"Finished updating estimates with bootstrapping.")

    def _update_conditional_estimates(self, t):
        self.logger.info(f"Starting updating estimates with bootstrapping.")
        t = np.asarray(t).reshape(-1)

        # initialize mean and var dictionaries if necessary
        if self._means_cond is None:
            self._means_cond =  {} # keys will be values of t
        if self._vars_cond is None:
            self._vars_cond = {} # keys will be values of t

        b = len(self.prediction_matrices)
        if b < 2:
            for _t in t:
                self._means_cond[_t] = 0 if b == 0 else (self.prediction_matrices[0]**2).mean(axis=0).sum()
                self._vars_cond[_t] = 0
            self.logger.info(f"Finished updating estimates with bootstrapping (since only one ensemble member is known, bootstrapping was skipped).")
            return
        
        matrices = np.array(self.prediction_matrices)

        means = []
        variances = []
        self.logger.debug(f"Now sampling {self.num_resamples} times {self.bootstrap_size} ensembles of size {max(t)}")
        for i in range(self.num_resamples):
            self.logger.debug(f"Creating %s-th bootstrap sample", i)

            # create random ensemble of the maximum given size
            ensemble_descriptors_through_indices = self.random_state.randint(0, b, size=(self.bootstrap_size, max(t)))
            ensemble_member_predictions = matrices[ensemble_descriptors_through_indices.ravel()].reshape(ensemble_descriptors_through_indices.shape + matrices.shape[1:])

            # compute errors for different sub-sizes of this ensemble on the given data
            means_for_round = []
            vars_for_round = []
            for size in t:
                ensemble_predictions = ensemble_member_predictions[:, :size, :, :].mean(axis=1)
                ensemble_errors = ((ensemble_predictions - self.y_oh)**2).mean(axis=1).sum(axis=1)
                means_for_round.append(ensemble_errors.mean())
                vars_for_round.append(ensemble_errors.var())
            means.append(means_for_round)
            variances.append(vars_for_round)
        
        for _t, mean_mean, mean_var in zip(t, np.mean(means, axis=0).reshape(t.shape[0]), np.mean(variances, axis=0).reshape(t.shape[0])):
            self._means_cond[_t] = mean_mean
            self._vars_cond[_t] = mean_var
        self.logger.info(f"Finished updating estimates with bootstrapping.")

    def estimate_performance_mean_in_iid_setup(self, t):
        self._check_param_coverage("E[Z_nt]")
        if not isinstance(t, (list, np.ndarray)):
            t = [t]
        if self._means_iid is None or any([_t not in self._means_iid for _t in t]) or not self.use_caching:
            self._update_iid_estimates(None, t)
        return np.array([self._means_iid[_t] for _t in t])

    def estimate_performance_mean_in_conditional_setup(self, t):
        self._check_param_coverage("E[Z_nt|D_val]")
        if not isinstance(t, (list, np.ndarray)):
            t = [t]
        if self._means_cond is None or any([_t not in self._means_cond for _t in t]) or not self.use_caching:
            self._update_conditional_estimates(t)
        return np.array([self._means_cond[_t] for _t in t])

    def estimate_performance_var_in_iid_setup(self, n, t):
        self._check_param_coverage("V[Z_nt]")
        if not isinstance(n, (list, np.ndarray)):
            n = [n]
        if not isinstance(t, (list, np.ndarray)):
            t = [t]
        if self._vars_iid is None or any([(_n, _t) not in self._vars_iid for _n in n for _t in t]) or not self.use_caching:
            self._update_iid_estimates(n, t)
        return np.array([[self._vars_iid[(_n, _t)] for _t in t] for _n in n])

    def estimate_performance_var_in_conditional_setup(self, t):
        self._check_param_coverage("V[Z_nt|D_val]")
        if not isinstance(t, (list, np.ndarray)):
            t = [t]
        if self._vars_cond is None or any([_t not in self._vars_cond for _t in t]) or not self.use_caching:
            self._update_conditional_estimates(t)
        return np.array([self._vars_cond[_t] for _t in t])
