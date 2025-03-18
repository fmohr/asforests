from .approach import Approach
import numpy as np
from scipy.stats import bootstrap


class DummyApproach(Approach):

    def __init__(self, random_state=None):
        super().__init__(random_state=random_state)
        self.prediction_matrices = None
    
    def reset(self):
        super().reset()
        self.prediction_matrices = []

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):

        # add prediction matrix
        self.prediction_matrices.append(prediction_matrix)

    def estimate_performance_mean_in_iid_setup(self, t):

        # how would we estimate i.i.d.?
        return self.estimate_performance_mean_in_conditional_setup(t)

    def estimate_performance_mean_in_conditional_setup(self, t):
        if isinstance(t, int):
            t = [t]
        
        means = []
        for size in t:
            mean_prediction = np.mean(self.prediction_matrices[:size], axis=0)
            mean_deviation = mean_prediction - self.y_oh
            means.append((mean_deviation**2).mean(axis=0).sum())
        return np.array(means)

    def estimate_performance_var_in_iid_setup(self, t):
        
        # how would we estimate i.i.d.?
        return self.estimate_performance_var_in_conditional_setup(t)

    def estimate_performance_var_in_conditional_setup(self, t):
        return np.zeros((len(t), ))