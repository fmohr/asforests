from abc import ABC, abstractmethod
import numpy as np


class Approach(ABC):

    def __init__(self, random_state=None):
        if random_state is None:
            random_state = np.random.RandomState()
        if isinstance(random_state, int):
            random_state = np.random.RandomState(random_state)
        self.seed = random_state.randint(low=0, high=10**7)
        self.random_state = None
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
    def estimate_performance_var_in_iid_setup(self, t):
        raise NotImplementedError
    
    @abstractmethod
    def estimate_performance_var_in_conditional_setup(self, t):
        raise NotImplementedError    


class TheoremBasedApproach(Approach, ABC):

    def __init__(self, random_state=None):
        super().__init__(random_state)

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

    def estimate_performance_var_in_iid_setup(self, t):

        # TODO: implement this
        return np.zeros((len(t), ))
    
    def estimate_performance_var_in_conditional_setup(self, t):
        # TODO: implement this
        return np.zeros((len(t), ))


class DeviationBasedApproach(TheoremBasedApproach):

    def __init__(self, random_state=None):
        super().__init__(random_state)

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        dev = prediction_matrix - self.y_oh
        self.receive_deviations_of_new_ensemble_member(dev)

    @abstractmethod
    def receive_deviations_of_new_ensemble_member(self, deviation_matrix):
        raise NotImplementedError