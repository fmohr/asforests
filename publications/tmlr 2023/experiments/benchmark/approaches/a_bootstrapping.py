from .approach import Approach
import numpy as np
from scipy.stats import bootstrap


class BootstrappingApproach(Approach):

    def __init__(self, num_resamples=10**3):
        self.prediction_matrices = []
        self.num_resamples = num_resamples

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):

        # add prediction matrix
        self.prediction_matrices.append(prediction_matrix)

    def estimate_performance_mean(self, t):

        if isinstance(t, int):
            t = [t]

        b = len(self.prediction_matrices)
        if b < 2:
            return np.zeros((len(t), ))
        
        rs = np.random.RandomState()
        matrices = np.array(self.prediction_matrices)
        means = []
        for size in t:
            ensemble_definitions = rs.choice(range(b), size=(self.num_resamples, size), replace=True)
            mean_for_size = []
            for ensemble in ensemble_definitions:
                mean_prediction = matrices[ensemble].mean(axis=0)
                mean_for_size.append(((mean_prediction - self.y_oh)**2).mean(axis=0).sum())
            means.append(np.mean(mean_for_size))
        return np.array(means)



    def estimate_performance_var(self, t):
        return np.zeros((len(t), ))
