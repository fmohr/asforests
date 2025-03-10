import numpy as np
from scipy.optimize import curve_fit

from .approach import TheoremBasedApproach


class ParametricModelApproach(TheoremBasedApproach):

    def __init__(self, rs=None, num_simulated_ensembles=1):

        # config
        if rs is None:
            rs = np.random.RandomState()
        self.rs = rs
        self.num_simulated_ensembles = num_simulated_ensembles

        # state
        self.prediction_matrices = []
        self._deviation_means = self._deviation_vars = self._deviation_covs = None

    @property
    def deviation_means(self):
        if self._deviation_means is None:
            self._update_estimates()
        assert self._deviation_means is not None
        return self._deviation_means

    @property
    def deviation_vars(self):
        if self._deviation_vars is None:
            self._update_estimates()
        assert self._deviation_vars is not None
        return self._deviation_vars

    @property
    def deviation_covs(self):
        if self._deviation_covs is None:
            self._update_estimates()
        assert self._deviation_covs is not None
        return self._deviation_covs

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        self.prediction_matrices.append(prediction_matrix)
        self._deviation_means = self._deviation_vars = self._deviation_covs = None

    def _update_estimates(self):

        # create permutations
        b = len(self.prediction_matrices)
        if b < 3:
            self._deviation_means = self._deviation_vars = self._deviation_covs = np.zeros(self.y_oh.shape[1])
            return
        ensembles = [self.rs.choice(range(b), size=b, replace=False) for i in range(self.num_simulated_ensembles)]

        # compute data for parametric learning problem
        sizes = []
        errors = []
        for ensemble in ensembles:
            ensemble_prediction_matrix = np.zeros(self.prediction_matrices[0].shape)
            for t, i in enumerate(ensemble, start=1):
                ensemble_prediction_matrix += (self.prediction_matrices[i] - ensemble_prediction_matrix) / t
                error_of_this_ensemble_per_target = ((ensemble_prediction_matrix - self.y_oh)**2).mean(axis=0)
                sizes.append(t)
                errors.append(error_of_this_ensemble_per_target)
        errors = np.array(errors).T

        # Define parametric function
        def model(t, a, b, c):
            return a**2 + b / t + (1 - 1/t) * c

        # estimate parameters
        self._deviation_means = []
        self._deviation_vars = []
        self._deviation_covs = []
        for j, target_errors in enumerate(errors):
            (a, b, c), covariance = curve_fit(model, sizes, target_errors)
            self._deviation_means.append(a)
            self._deviation_vars.append(b)
            self._deviation_covs.append(c)
        self._deviation_means = np.array(self._deviation_means)
        self._deviation_vars = np.array(self._deviation_vars)
        self._deviation_covs = np.array(self._deviation_covs)

