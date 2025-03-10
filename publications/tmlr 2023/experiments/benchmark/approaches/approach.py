from abc import ABC, abstractmethod
import numpy as np


class Approach(ABC):

    def __init__(self):
        self.y_oh = None

    def tell_ground_truth_labels(self, y_oh):
        self.y_oh = y_oh

    @abstractmethod
    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        raise NotImplementedError

    @abstractmethod
    def estimate_performance_mean(self, t):
        raise NotImplementedError

    @abstractmethod
    def estimate_performance_var(self, t):
        raise NotImplementedError


class TheoremBasedApproach(Approach, ABC):

    @property
    @abstractmethod
    def deviation_means(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def deviation_vars(self):
        raise NotImplementedError

    @property
    @abstractmethod
    def deviation_covs(self):
        raise NotImplementedError

    def estimate_performance_mean(self, t):
        if isinstance(t, list):
            t = np.array(t)
        return np.sum(self.deviation_means**2) + np.sum(self.deviation_vars) / t + (1 - 1/t) * np.sum(self.deviation_covs)

    def estimate_performance_var(self, t):

        # TODO: implement this
        return np.zeros((len(t), ))


class DeviationBasedApproach(TheoremBasedApproach):

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        dev = prediction_matrix - self.y_oh
        self.receive_deviations_of_new_ensemble_member(dev)

    @abstractmethod
    def receive_deviations_of_new_ensemble_member(self, deviation_matrix):
        raise NotImplementedError