import numpy as np
import pandas as pd
from numpy.linalg import lstsq

from .approach import Approach

import logging
from tqdm import tqdm


class ParametricDifferenceModelApproach(Approach):

    def __init__(
            self,
            num_simulated_ensembles=1,
            with_replacement=False,
            anchors="power",
            show_progress=False,
            **kwargs
            ):
        super().__init__(**kwargs)

        # config
        self.anchors = anchors
        self.num_simulated_ensembles = num_simulated_ensembles
        self.with_replacement = with_replacement
        self.show_progress = show_progress

        # state variables
        self._sample_of_ensemble_performances = None
        self._p_mean = None
        self._p_cvar = None

    
    def reset(self):

        # state
        super().reset()
        self.prediction_matrices = []
        self._sample_of_ensemble_performances = None
        self._p_mean = self._p_cvar = self._p_iidvar = None

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        self.prediction_matrices.append(prediction_matrix)
        self._sample_of_ensemble_performances = None
        self._p_mean = self._p_cvar = self._p_iidvar = None
        self.logger.info(f"Added prediction matrix #{len(self.prediction_matrices)}")
    
    
    def _get_schedule_for_max_anchor(self, max_anchor=10**3):
        """
           :param max_anchor: this variable define up to which t we want to create data points. Could be `b` or some constant
        """
        return sorted(set([int(np.round(2**(i / 2))) for i in range(int(2 * np.log2(max_anchor) + 1))]))

    def _sample_ensemble_performances_at_schedule(self):
        b = len(self.prediction_matrices)
        max_anchor = 10**3
        self.logger.info(f"Drawing {self.num_simulated_ensembles} ensembles of size {b}")
        ensembles = self.random_state.randint(0, b, size=(max_anchor, self.num_simulated_ensembles))
        schedule = self._get_schedule_for_max_anchor(max_anchor)

        # compute data for parametric learning problem
        sizes = []
        errors = []
        self.logger.info(f"Computing database")
        if self.anchors == "full":
            for ensemble in ensembles:

                ensemble_prediction_matrix = np.zeros(self.prediction_matrices[0].shape)
                for s, i in enumerate(ensemble, start=1):
                    ensemble_prediction_matrix += (self.prediction_matrices[i] - ensemble_prediction_matrix) / s
                    error_of_this_ensemble = ((ensemble_prediction_matrix - self.y_oh)**2).mean(axis=0).sum()
                    sizes.append(s)
                    errors.append(error_of_this_ensemble)
            errors = np.array(errors)
            
        elif self.anchors.startswith("power"):
            matrices = np.array(self.prediction_matrices)
            for size in schedule:
                ensemble_member_predictions = matrices[ensembles[:size].ravel()].reshape((size, ensembles.shape[1]) + matrices.shape[1:])
                ensemble_prediction_matrix = ensemble_member_predictions.mean(axis=0)
                errors_for_this_size = ((ensemble_prediction_matrix - self.y_oh)**2).mean(axis=1).sum(axis=1)
                added_errors = errors_for_this_size.ravel()
                sizes.extend(len(added_errors) * [size])
                errors.extend(added_errors)
        self._sample_of_ensemble_performances = pd.DataFrame(data={"t": sizes, "Z_nt": errors})
        self.logger.info(f"Done. Database has {len(self._sample_of_ensemble_performances)} entries.")

    
    def _estimate_params_for_mean(self):
        # if we do not have enough observations, return 0
        # TODO: return empirical mean
        b = len(self.prediction_matrices)
        if b < 2:
            self._p_mean = np.zeros(2)
            return
        
        # create permutations
        self._sample_ensemble_performances_at_schedule()

        # estimate parameters for mean
        self.logger.info(f"Now fitting the model.")
        X = np.column_stack((np.ones(len(self._sample_of_ensemble_performances)), 1 / np.array(self._sample_of_ensemble_performances["t"])))
        self._p_mean = lstsq(X, self._sample_of_ensemble_performances["Z_nt"])[0]
        self.logger.info(f"Done, stored values {self._p_mean}.")
    
    def _estimate_params_for_conditional_var(self):
        # if we do not have enough observations, return 0
        # TODO: return empirical mean
        b = len(self.prediction_matrices)
        if b < 2:
            self._p_cvar = np.zeros(4)
            return
        
        # this variable define up to which t we want to create data points. Could be `b` or some constant
        max_anchor = 10**3
        
        # estimate E[Z_nt] at all anchors in the schedule
        means_at_schedule_points = {
            t: self.estimate_performance_mean_in_conditional_setup(t=[t])[0]
            for t in self._get_schedule_for_max_anchor(max_anchor)
        }
        
        # the ensemble performances at the schedule were implicitly computed, so we can now use them to define the targets
        sizes = []
        targets = []
        for t, df_t in self._sample_of_ensemble_performances.groupby("t"):
            mu = means_at_schedule_points[t]
            sizes.extend([t] * len(df_t))
            targets.extend((mu - df_t["Z_nt"])**2)

        # estimate parameters for mean
        self.logger.info(f"Now fitting {len(targets)} models, one per target.")
        X = np.column_stack((np.ones_like(sizes), 1 / np.array(sizes), 1 / np.array(sizes)**2, 1 / np.array(sizes)**3))
        self._p_cvar = lstsq(X, targets)[0]
        self.logger.info("Successfully fitted model for the variance.")

    def _estimate_params_for_iid_var(self):
        # if we do not have enough observations, return 0
        # TODO: return empirical mean
        b = len(self.prediction_matrices)
        if b < 7:
            self._p_iidvar  = np.zeros(7)
            return
        
        # this variable define up to which t we want to create data points. Could be `b` or some constant
        max_anchor = 10**3
        
        # estimate E[Z_nt] at all anchors in the schedule
        means_at_schedule_points = {
            t: self.estimate_performance_mean_in_iid_setup(t=[t])[0]
            for t in self._get_schedule_for_max_anchor(max_anchor)
        }
        
        # the ensemble performances at the schedule were implicitly computed, so we can now use them to define the targets
        sizes = []
        targets = []
        for t, df_t in self._sample_of_ensemble_performances.groupby("t"):
            mu = means_at_schedule_points[t]
            sizes.extend([t] * len(df_t))
            targets.extend((mu - df_t["Z_nt"])**2)
        n = self.prediction_matrices[0].shape[0]
        sizes = np.array(sizes)

        # estimate parameters for mean
        self.logger.info(f"Now fitting {len(targets)} models, one per target.")
        X = np.column_stack((np.ones_like(sizes) / n, 1 / (n * sizes), 1 / (n * sizes**2), 1 / (n * sizes**3), 1 / sizes, 1 / sizes**2, 1 / sizes**3))
        self._p_iidvar = lstsq(X, targets)[0]
        self.logger.info("Successfully fitted model for the variance.")

    def estimate_performance_mean_in_iid_setup(self, t):
        t = np.asarray(t).reshape(-1)
        if self._p_mean is None:
            self._estimate_params_for_mean()
        return self._p_mean[0] + self._p_mean[1] / t
    
    def estimate_performance_mean_in_conditional_setup(self, t):
        t = np.asarray(t).reshape(-1)
        if self._p_mean is None:
            self._estimate_params_for_mean()
        return self._p_mean[0] + self._p_mean[1] / t

    def estimate_performance_var_in_iid_setup(self, n, t):
        n = np.asarray(n).reshape(-1)
        t = np.asarray(t).reshape(-1)
        if self._p_iidvar is None:
            self._estimate_params_for_iid_var()
        if not isinstance(t, np.ndarray):
            if not isinstance(t, list):
                t = [t]
            t = np.array(t)
        if not isinstance(n, np.ndarray):
            if not isinstance(n, list):
                n = [n]
            n = np.array(n)
        return np.maximum(0, np.array(
            [
                self._p_iidvar[0] / _n +
                self._p_iidvar[1] / (_n * t) +
                self._p_iidvar[2] / (_n * t**2) +
                self._p_iidvar[3] / (_n * t**3) +
                self._p_iidvar[4] / t +
                self._p_iidvar[5] / t**2 +
                self._p_iidvar[6] / t**3
                for _n in n
            ]).reshape((len(n), len(t)))) # make sure to not return negative values
    
    def estimate_performance_var_in_conditional_setup(self, t):
        t = np.asarray(t).reshape(-1)
        if self._p_cvar is None:
            self._estimate_params_for_conditional_var()
        return np.maximum(0, self._p_cvar[0] + self._p_cvar[1] / t + self._p_cvar[2] / t**2 + self._p_cvar[3] / t**3) # make sure to not return negative values
        
