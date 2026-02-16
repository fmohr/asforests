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
    def estimate_performance_var_in_iid_setup(self, n, t): # this is the only parameter where the n is given explicitly since in all other scenarios, it is either irrelevant or implicit by the given data
        raise NotImplementedError
    
    @abstractmethod
    def estimate_performance_var_in_conditional_setup(self, t):
        raise NotImplementedError    


class TheoremBasedApproach(Approach, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    @abstractmethod
    def n_validation(self):
        raise NotImplementedError

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
        t = np.asarray(t).reshape(-1)
        return np.array([
            np.ones(t.shape[0]),
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
        c = np.concatenate([c1, c2], axis=0)
        c[10] = c[11] = c[13] = 0 # by theory, we know that these coefficients must be 0
        return c

    def estimate_performance_mean_in_iid_setup(self, t):
        t = np.asarray(t).reshape(-1)
        return np.sum(self.deviation_means_in_iid_setting**2) + np.sum(self.deviation_vars_in_iid_setting) / t + (1 - 1/t) * np.sum(self.deviation_covs_in_iid_setting)
    
    def estimate_performance_mean_in_conditional_setup(self, t):
        """
            Here we can exploit the fact that, conditioned on specific data, the variances becomes independent across ensemble members
        """
        t = np.asarray(t).reshape(-1)
        if self.deviation_means_in_conditional_setting is None:
            raise ValueError(f"deviation_means_in_conditional_setting is None for {self.__class__}")
        if self.deviation_means_in_conditional_setting.shape != self.y_oh.shape:
            raise ValueError(f"deviation_means_in_conditional_setting has wrong shape for {self.__class__}. Should be {self.y_oh.shape} but is {self.deviation_means_in_conditional_setting.shape}")
        return (self.deviation_means_in_conditional_setting**2).mean(axis=0).sum() + self.deviation_vars_in_conditional_setting.mean(axis=0).sum() / t

    def estimate_performance_var_in_iid_setup(self, n, t):
        n = np.asarray(n).reshape(-1)
        t = np.asarray(t).reshape(-1)
        coeffiecients_for_different_t = self.get_xi_cov_coefficients_for_iid_scenario(t)
        assert coeffiecients_for_different_t.shape == (14, len(t)), f"Incorrect shape for xi-coefficients. Should be {(14, len(t))} but is {coeffiecients_for_different_t.shape}"
        coeffiecients_for_different_t[10] = coeffiecients_for_different_t[11] = coeffiecients_for_different_t[13] = 0 # by theory, we know that these coefficients must be 0
        out= []
        for _n in n:
            out_for_n = []
            for coefficients_for_t, _t in zip(coeffiecients_for_different_t.T, t):
                weighted_cov_summands = coefficients_for_t * self.xi_covs_in_iid_setting
                out_for_n.append(np.sum((weighted_cov_summands[:7] / _n + weighted_cov_summands[7:] * (_n - 1) / _n)) / _t**3)
            out.append(out_for_n)
        out = np.array(out)
        return out
    
    def estimate_performance_var_in_conditional_setup(self, t):
        t = np.asarray(t).reshape(-1)
        self.logger.info(f"Computing estimate of V[Z_nt|D_val] for {t=}.")
        coeffiecients = self.get_xi_cov_coefficients_for_conditional_scenario(t)
        covs = self.xi_covs_in_conditional_setting
        if not isinstance(covs, np.ndarray):
            raise ValueError(f"xi_covs_in_conditional_setting must return a numpy array with 7 entries but is of type {type(covs)}")
        if (7, ) != covs.shape:
            raise ValueError(f"xi_covs_in_conditional_setting must return a vector with 7 entries but has shape {covs.shape}")
        return coeffiecients.T @ covs / t**3


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