import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from .approach import TheoremBasedApproach


class ParametricModelApproach(TheoremBasedApproach):

    def __init__(
            self,
            num_simulated_ensembles=1,
            with_replacement=False,
            random_state=None
            ):
        super().__init__(random_state=random_state)

        # config
        self.num_simulated_ensembles = num_simulated_ensembles
        self.with_replacement = with_replacement
        self._instance_wise_deviation_means = self._instance_wise_deviation_vars = None  # variables for conditional setting
        self._deviation_means = self._deviation_vars = self._deviation_covs = None  # variables for iid setting
    
    def reset(self):

        # state
        super().reset()
        self.prediction_matrices = []
        self._deviation_means = self._deviation_vars = self._deviation_covs = None

    @property
    def deviation_means_in_iid_setting(self):
        if self._deviation_means is None:
            self._update_estimates_for_iid()
        assert self._deviation_means is not None
        return self._deviation_means
    
    @property
    def deviation_means_in_conditional_setting(self):
        if self._instance_wise_deviation_means is None:
            self._update_estimates_for_conditional()
        assert self._instance_wise_deviation_means is not None
        return self._instance_wise_deviation_means

    @property
    def deviation_vars_in_iid_setting(self):
        if self._deviation_vars is None:
            self._update_estimates_for_iid()
        assert self._deviation_vars is not None
        return self._deviation_vars

    @property
    def deviation_vars_in_conditional_setting(self):
        if self._instance_wise_deviation_vars is None:
            self._update_estimates_for_conditional()
        assert self._instance_wise_deviation_vars is not None
        return self._instance_wise_deviation_vars

    @property
    def deviation_covs_in_iid_setting(self):
        if self._deviation_covs is None:
            self._update_estimates()
        assert self._deviation_covs is not None
        return self._deviation_covs

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        self.prediction_matrices.append(prediction_matrix)
        self._deviation_means = self._deviation_vars = self._deviation_covs = None  # this will enforce re-computation the next time
        self._instance_wise_deviation_means = self._instance_wise_deviation_vars = None  # this will enforce re-computation the next time

    def _update_estimates_for_iid(self):

        # create permutations
        b = len(self.prediction_matrices)
        if b < 3:
            self._deviation_means = self._deviation_vars = self._deviation_covs = np.zeros(self.y_oh.shape[1])
            return
        ensembles = [self.random_state.choice(range(b), size=b, replace=self.with_replacement) for _ in range(self.num_simulated_ensembles)]

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

    def _update_estimates_for_conditional(self):

        """
            In this case, we build a model for each instance/target combination

            This is because we know that E[Z_nt|D_val] can be decomposed into E[Z_ijt], where i is a specific instance and j the target.
        """

        # create permutations
        budget = len(self.prediction_matrices)
        if budget < 3:
            self._instance_wise_deviation_means = self._instance_wise_deviation_vars = np.zeros(self.y_oh.shape)
            return
        ensembles = [self.random_state.choice(range(budget), size=budget, replace=self.with_replacement) for _ in range(self.num_simulated_ensembles)]

        # compute for different ensembles of different sizes the left hand side
        errors = np.zeros((len(ensembles), budget, self.y_oh.shape[0],self.y_oh.shape[1]))
        for e_id, ensemble in enumerate(ensembles):
            ensemble_prediction_matrix = np.zeros(self.prediction_matrices[0].shape)
            for t in range(1, budget + 1):
                ensemble_prediction_matrix += (self.prediction_matrices[ensemble[t - 1]] - ensemble_prediction_matrix) / t
                errors[e_id, t-1] = (ensemble_prediction_matrix - self.y_oh)**2
        
        # compile datasets for each instance/target combination
        prediction_goals = errors.transpose(2, 3, 0, 1)
        datasets = {
            (i, j): pd.DataFrame({"t": self.num_simulated_ensembles * list(range(1, budget + 1)), "y": prediction_goals[i, j].flatten()})
            for i in range(self.y_oh.shape[0])
            for j in range(self.y_oh.shape[1])
        }

        # Define parametric function
        def model(t, a, b):
            return a**2 + b / t

        # estimate parameters
        self._instance_wise_deviation_means = np.zeros(self.y_oh.shape)
        self._instance_wise_deviation_vars = np.zeros(self.y_oh.shape)
        
        for (i, j), dataset in datasets.items():
            if dataset["y"].min() == dataset["y"].max():
                a, b = dataset["y"].min(), 0
            else:
                (a, b), _ = curve_fit(model, dataset["t"], dataset["y"])
                if False and budget > 8:
                    import matplotlib.pyplot as plt
                    fig, ax = plt.subplots()
                    ax.scatter(dataset["t"], dataset["y"], alpha=0.5)
                    domain = np.arange(1, budget + 1)
                    ax.plot(domain, a**2 + b / domain)
                    plt.show()
            self._instance_wise_deviation_means[i, j] = a
            self._instance_wise_deviation_vars[i, j] = b
        
