from .approach import Approach
import numpy as np
from scipy.stats import bootstrap


class BootstrappingApproach(Approach):

    def __init__(self, num_resamples=10**3, random_state=None):
        super().__init__(random_state=random_state)
        self.prediction_matrices = None
        self.num_resamples = num_resamples
    
    def reset(self):
        super().reset()
        self.prediction_matrices = []

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):

        # add prediction matrix
        self.prediction_matrices.append(prediction_matrix)

    def estimate_performance_mean_in_iid_setup(self, t):

        # i.i.d. estimate even possible???
        return self.estimate_performance_mean_in_conditional_setup(t)

    def estimate_performance_mean_in_conditional_setup(self, t):
        if isinstance(t, int):
            t = [t]

        b = len(self.prediction_matrices)
        if b < 2:
            return np.zeros((len(t), ))
        
        matrices = np.array(self.prediction_matrices)
        means = []
        for size in t:
            ensemble_definitions = self.random_state.choice(range(b), size=(self.num_resamples, size), replace=True)
            mean_for_size = []
            for ensemble in ensemble_definitions:
                mean_prediction = matrices[ensemble].mean(axis=0)
                mean_for_size.append(((mean_prediction - self.y_oh)**2).mean(axis=0).sum())
            means.append(np.mean(mean_for_size))
        return np.array(means)

    def estimate_performance_var_in_iid_setup(self, t):

        # i.i.d. estimate even possible???
        return self.estimate_performance_var_in_conditional_setup(t)

    def estimate_performance_var_in_conditional_setup(self, t):
        return np.zeros((len(t), ))