from abc import ABC, abstractmethod
import numpy as np
import logging


class Approach(ABC):

    def __init__(self, estimated_parameters=None, random_state=None, logger=None):
        self.estimated_parameters = ["E[Z_nt]", "E[Z_nt|D_val]", "V[Z_nt]", "V[Z_nt|D_val]"] if estimated_parameters is None else estimated_parameters
        if random_state is None:
            random_state = np.random.RandomState()
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        self.seed = random_state.randint(low=0, high=10**7)
        self.random_state = None
        self.logger = logger if logger is not None else logging.getLogger("approach")
        self.y_oh = None
    
    def reset(self):
        self.y_oh = None
        self.random_state = np.random.RandomState(self.seed)

    def tell_ground_truth_labels(self, y_oh):
        self.y_oh = y_oh

    @abstractmethod
    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        raise NotImplementedError

    @abstractmethod
    def estimate_performance_mean_in_iid_setup(self, t):
        raise NotImplementedError
    
    @abstractmethod
    def estimate_performance_mean_in_conditional_setup(self, t):
        raise NotImplementedError    

    @abstractmethod
    def estimate_performance_var_for_two_instances_in_iid_setup(self, t):
        raise NotImplementedError
    
    @abstractmethod
    def estimate_performance_var_in_conditional_setup(self, t, n):
        """
            The n here is only a control parameter, because it could also be inferred from the prediction/deviation matrix size (which must coincide in this)
        """
        raise NotImplementedError    


class TheoremBasedApproach(Approach, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def deviation_means_in_iid_setting(self):
        """
            a vector of size k with one estimate of the mean deviation (across i.i.d. sampled instances and ensemble members) for each target
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def deviation_means_in_conditional_setting(self):
        """
            an n x k matrix with an estimate of the mean deviations of (i.i.d. sampled) ensemble members on n given validation instances
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def deviation_vars_in_iid_setting(self):
        """
            a vector of size k with one estimate of the variance in deviation (across i.i.d. sampled instances and ensemble members) for each target
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def deviation_vars_in_conditional_setting(self):
        """
            an n x k matrix with an estimate of the variance of deviations of (i.i.d. sampled) ensemble members on n given validation instances
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def deviation_covs_in_iid_setting(self):
        raise NotImplementedError
    
    @property
    @abstractmethod
    def xi_covs_in_iid_setting(self):
        """
            a vector of 14 entries. The first 7 are the covariances of xi-terms for identical instances. The remaining are covs of xi-terms for deviating instances.
        """
        raise NotImplementedError
    
    @property
    @abstractmethod
    def xi_covs_in_conditional_setting(self):
        """
            a 3D tensor of shape (n, n, 7), where n is the number of validation instances. The entry [i1, i2, j] has the j-th xi-covariance term between instance i1 and i2
        """
        raise NotImplementedError

    def get_xi_cov_coefficients_for_conditional_scenario(self, t):
        return np.array([
            np.ones(len(t)),
            (t-1) * 2,
            (t-1) * 4,
            (t-1),
            (t-1)*(t-2)*2,
            (t-1)*(t-2)*4,
            (t-1)*(t-2)*(t-3)
        ])
    
    def get_xi_cov_coefficients_for_iid_scenario(self, t):
        c1 = self.get_xi_cov_coefficients_for_conditional_scenario(t)
        c2 = c1
        return np.concatenate([c1, c2], axis=0)

    def estimate_performance_mean_in_iid_setup(self, t):
        if isinstance(t, list):
            t = np.array(t)
        return np.sum(self.deviation_means_in_iid_setting**2) + np.sum(self.deviation_vars_in_iid_setting) / t + (1 - 1/t) * np.sum(self.deviation_covs_in_iid_setting)
    
    def estimate_performance_mean_in_conditional_setup(self, t):
        """
            Here we can exploit the fact that, conditioned on specific data, the variances becomes independent across ensemble members
        """
        if isinstance(t, list):
            t = np.array(t)
        if self.deviation_means_in_conditional_setting is None:
            raise ValueError(f"deviation_means_in_conditional_setting is None for {self.__class__}")
        if self.deviation_means_in_conditional_setting.shape != self.y_oh.shape:
            raise ValueError(f"deviation_means_in_conditional_setting has wrong shape for {self.__class__}. Should be {self.y_oh.shape} but is {self.deviation_means_in_conditional_setting.shape}")
        return (self.deviation_means_in_conditional_setting**2).mean(axis=0).sum() + self.deviation_vars_in_conditional_setting.mean(axis=0).sum() / t

    def estimate_performance_var_for_two_instances_in_iid_setup(self, t):
        coeffiecients = self.get_xi_cov_coefficients_for_iid_scenario(t)
        coeffiecients[10] = coeffiecients[11] = coeffiecients[13] = 0 # by theory, we know that these coefficients must be 0
        sum_of_covs = coeffiecients.T @ self.xi_covs_in_iid_setting
        return sum_of_covs / (2 * t**3) # divide by 2 since this is our n here (this applies to all terms, because also (n - 1) / n = 1 / 2 for n = 2)
    
    def estimate_performance_var_in_conditional_setup(self, t):
        self.logger.info(f"Computing estimate of V[Z_nt|D_val] for {t=}.")
        coeffiecients = self.get_xi_cov_coefficients_for_conditional_scenario(t)
        covs = self.xi_covs_in_conditional_setting
        if not isinstance(covs, np.ndarray):
            raise ValueError(f"xi_covs_in_conditional_setting must return a n x n x 7 numpy array but is of type {type(covs)}")
        if len(covs.shape) != 3 or covs.shape[0] != covs.shape[1] or covs.shape[2] != 7:
            raise ValueError(f"xi_covs_in_conditional_setting must return a n x n x 7 tensor but has shape {covs.shape}")
        sum_of_terms = np.einsum("ijk,kt->t", covs, coeffiecients)
        return sum_of_terms / (covs.shape[0]**2 * t**3)


class DeviationBasedApproach(TheoremBasedApproach):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        if self.y_oh is None:
            raise RuntimeError(f"Ground truth targets not initialized. Check whether you called tell_ground_truth_labels.")
        dev = prediction_matrix - self.y_oh
        self.receive_deviations_of_new_ensemble_member(dev)

    @abstractmethod
    def receive_deviations_of_new_ensemble_member(self, deviation_matrix):
        raise NotImplementedError