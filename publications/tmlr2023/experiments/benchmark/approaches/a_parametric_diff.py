import numpy as np
import pandas as pd
from numpy.linalg import lstsq
import itertools as it

from .approach import Approach

import logging
from tqdm import tqdm
import time

from sklearn.linear_model import Ridge

import matplotlib.pyplot as plt

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
        self._sample_of_ensemble_performances_iid = None
        self._sample_of_ensemble_performances_cond = None
        self._p_mean_iid = None
        self._p_mean_cond = None
        self.model_v_iid = None
        self.model_v_cond = None
    
    def reset(self):

        # state
        super().reset()
        self.deviation_matrices = []
        self._sample_of_ensemble_performances = None
        self._p_mean_cond = self._p_cvar_cond = self._p_iidvar = None

    def receive_predictions_of_new_ensemble_member(self, prediction_matrix):
        self.deviation_matrices.append(prediction_matrix - self.y_oh)
        self._sample_of_ensemble_performances = None
        self._p_mean_cond = self._p_mean_iid = self._p_cvar_cond = self._p_iidvar = None
        self.model_v_iid = None
        self.model_v_cond = None
        self.logger.info(f"Added deviation matrix #{len(self.deviation_matrices)} of shape {prediction_matrix.shape}")
    
    
    def _get_schedule_for_max_anchor(self, max_anchor=10**3):
        """
           :param max_anchor: this variable define up to which t we want to create data points. Could be `b` or some constant
        """
        schedule = sorted(set([int(np.round(2**(i / 4))) for i in range(int(4 * np.log2(max_anchor) + 1))]))
        if max_anchor not in schedule:
            schedule.append(max_anchor)
        return np.array(schedule)

    def _sample_ensemble_performances_at_anchors(self, iid_data, n, t, num_samples):

        deviation_matrices = np.array(self.deviation_matrices)
        b = len(deviation_matrices)
        
        # to compute the ensemble behaviors, we usually need matrices that don't fit into memory, so we compute them in batches
        required_entries = t * num_samples * (n if iid_data else deviation_matrices.shape[1]) * deviation_matrices.shape[2]
        MAX_ENTRIES_IN_MATRIX = 10**8
        max_batch_size = int(np.ceil(MAX_ENTRIES_IN_MATRIX / (t * (n if iid_data else deviation_matrices.shape[1]) * deviation_matrices.shape[2])))
        num_batches = int(np.ceil(required_entries / MAX_ENTRIES_IN_MATRIX))
        num_remaining_entries = num_samples
        self.logger.debug(f"Computing {num_samples} ensemble prediction matrices at anchor {n}, {t} in {num_batches} batches; {iid_data=}.")
        
        # get matrices with ensemble predictions on validation or iid sampled data (factorizing out ensemble member behavior)
        ensemble_deviation_matrix_collection = []
        for _ in range(num_batches):

            # first get prediction matrices for all ensemble members of all ensembles in the current batch
            actual_batch_size = min(num_remaining_entries, max_batch_size)
            ensemble_members_in_batch = self.random_state.randint(0, b, size=t * actual_batch_size)
            ensemble_member_deviations = deviation_matrices[ensemble_members_in_batch].reshape((t, actual_batch_size) + deviation_matrices.shape[1:]) # shape: (t, actual_batch_size, n, k)

            # in the case of iid data, the prediction matrices should not be over the validation data but over randomly sampled instances
            if iid_data:
                instance_indices_in_batch = self.random_state.randint(0, deviation_matrices.shape[1], size=(actual_batch_size, n))
                instance_indices_in_batch = instance_indices_in_batch[None, :, :, None]
                ensemble_member_deviations = np.take_along_axis(ensemble_member_deviations, instance_indices_in_batch, axis=2)
            
            # check the shape of the ensemble member predictions
            assert ensemble_member_deviations.shape == (t, actual_batch_size, deviation_matrices.shape[1] if not iid_data else n, deviation_matrices.shape[2]), f"Expected shape {(t, actual_batch_size, deviation_matrices.shape[1], deviation_matrices.shape[2])}, got {ensemble_member_deviations.shape}."
            
            # get ensemble deviations by aggregating the deviations across members
            ensemble_deviation_matrix_collection.append(ensemble_member_deviations.mean(axis=0))
            num_remaining_entries -= actual_batch_size
        ensemble_deviation_matrices = np.concatenate(ensemble_deviation_matrix_collection)
        self.logger.debug(f"Computed {actual_batch_size} ensemble prediction matrices at anchor {t} of shape {ensemble_deviation_matrices.shape}.")
        
        # compute the errors for each ensemble
        errors_for_this_size = (ensemble_deviation_matrices**2).mean(axis=1).sum(axis=1)
        assert errors_for_this_size.shape == (num_samples, ), f"Expected shape {(num_samples, )}, got {errors_for_this_size.shape}."
        return errors_for_this_size

    def _sample_ensemble_performances_at_schedule(self, iid_data: bool):
        b = len(self.deviation_matrices)
        max_anchor_for_t = max(10**3, b)
        max_anchor_for_n = 32
        schedule_for_t = self._get_schedule_for_max_anchor(max_anchor_for_t)
        schedule_for_n = self._get_schedule_for_max_anchor(max_anchor_for_n)

        # compute data for parametric learning problem
        num_samples_per_case = self.num_simulated_ensembles / len(schedule_for_t)
        if iid_data and "V[Z_nt]" in self.estimated_parameters:
            num_samples_per_case /= len(schedule_for_n)
        num_samples_per_case = max(1, int(num_samples_per_case))
        max_index_for_overhead = self.num_simulated_ensembles - num_samples_per_case * len(schedule_for_t) * (len(schedule_for_n) if iid_data and "V[Z_nt]" in self.estimated_parameters else 1)
        
        t_vals = []
        n_vals = []
        errors = []
        self.logger.info(f"Computing database. Using geometric schedule with {len(schedule_for_t)} anchors for t {schedule_for_t}.")
        i = 0
        for t in schedule_for_t: # create data for each anchor on the schedule. The anchor is used for both ensemble size and number of application instances (relevant for estimate of V[Z_nt])
            if i >= self.num_simulated_ensembles:
                break
            if iid_data:
                if "V[Z_nt]" in self.estimated_parameters:
                    for n in schedule_for_n:
                        if i >= self.num_simulated_ensembles:
                            break
                        errors_for_this_size = self._sample_ensemble_performances_at_anchors(iid_data=True, n=n, t=t, num_samples=num_samples_per_case + (1 if i < max_index_for_overhead else 0))
                        t_vals.extend(len(errors_for_this_size) * [t])
                        n_vals.extend(len(errors_for_this_size) * [n])
                        errors.extend(errors_for_this_size)
                        i += 1
                else:
                    errors_for_this_size = self._sample_ensemble_performances_at_anchors(iid_data=True, n=1, t=t, num_samples=num_samples_per_case + (1 if i < max_index_for_overhead else 0))
                    t_vals.extend(len(errors_for_this_size) * [t])
                    n_vals.extend(len(errors_for_this_size) * [1])
                    errors.extend(errors_for_this_size)
                    i += 1
            else:
                errors_for_this_size = self._sample_ensemble_performances_at_anchors(iid_data=False, n=None, t=t, num_samples=num_samples_per_case + (1 if i < max_index_for_overhead else 0))
                t_vals.extend(len(errors_for_this_size) * [t])
                n_vals.extend(len(errors_for_this_size) * [None])
                errors.extend(errors_for_this_size)
                i += 1
        
        # store the observed performances in the respective dataframe
        df = pd.DataFrame(data={"t": t_vals, "n": n_vals, "Z_nt": errors})
        self.logger.info(f"Done. Database has {len(df)} entries.")
        if iid_data:
            self._sample_of_ensemble_performances_iid = df
        else:
            self._sample_of_ensemble_performances_cond = df

    
    def _estimate_params_for_mean(self, iid_data: bool):
        
        # create permutations
        self._sample_ensemble_performances_at_schedule(iid_data)

        # estimate parameters for mean
        self.logger.info(f"Now fitting the model.")
        sampled_performances = self._sample_of_ensemble_performances_iid if iid_data else self._sample_of_ensemble_performances_cond
        X = np.column_stack((np.ones(len(sampled_performances)), 1 / np.array(sampled_performances["t"])))
        p = lstsq(X, sampled_performances["Z_nt"])[0]
        if iid_data:
            self._p_mean_iid = p
        else:
            self._p_mean_cond = p
        self.logger.info(f"Done, stored values {self._p_mean_cond}.")
    
    def _estimate_params_for_conditional_var(self):
        
        # get mean performances at each t in the schedule
        b = len(self.deviation_matrices)
        schedule = self._get_schedule_for_max_anchor(max(10**3, b))
        means = self.estimate_performance_mean_in_conditional_setup(t=schedule)

        # the ensemble performances at the schedule were implicitly computed, so we can now use them to define the targets
        sizes = []
        targets = []
        for (t, df_t), mean in zip(self._sample_of_ensemble_performances_cond.groupby("t"), means):
            sizes.extend([t] * len(df_t))
            targets.extend((mean - df_t["Z_nt"])**2)
        sizes = np.array(sizes)
        targets = np.array(targets)

        # estimate parameters for mean
        self.logger.info(f"Now fitting a model with {len(targets)} data points.")
        X = np.column_stack((np.ones_like(sizes), 1 / np.array(sizes), 1 / np.array(sizes)**2, 1 / np.array(sizes)**3))
        self.model_v_cond = Ridge(alpha=0.0)
        self.model_v_cond.fit(X, targets, sample_weight=sizes)#1 / (targets**2 + 10**-10))
        self.logger.info(f"Model for parameters of V[Z_nt|D_val] ready.")

    def _estimate_params_for_iid_var(self, n):

        # get mean performances at each t in the schedule
        b = len(self.deviation_matrices)
        schedule = self._get_schedule_for_max_anchor(max(10**3, b))
        means = self.estimate_performance_mean_in_iid_setup(t=schedule)

        # use two loops to gather data, because means are invariant to n and should only be paired with t
        vals_t = []
        vals_n = []
        targets = []
        for (t, df_t), mean in zip(self._sample_of_ensemble_performances_iid.groupby("t"), means):
            for n, df_nt in df_t.groupby("n"):
                vals_t.extend([t] * len(df_nt))
                vals_n.extend([n] * len(df_nt))
                targets.extend((mean - df_nt["Z_nt"])**2)
        vals_t = np.array(vals_t)
        vals_n = np.array(vals_n)
        targets = np.array(targets)

        # estimate parameters for mean
        self.logger.info(f"Now fitting a model with {len(targets)} datapoints.")
        X = np.column_stack((1 / vals_n, 1 / (vals_n * vals_t), 1 / (vals_n * vals_t**2), 1 / (vals_n * vals_t**3), 1 / vals_t, 1 / vals_t**2, 1 / vals_t**3))
        self.model_v_iid = Ridge(alpha=0.0)
        self.model_v_iid.fit(X, targets)#, sample_weight=vals_t)
        self.logger.info("Successfully fitted model for the variance.")

    def estimate_performance_mean_in_iid_setup(self, t):
        t = np.asarray(t).reshape(-1)
        if self._p_mean_iid is None:
            self._estimate_params_for_mean(iid_data=True)
        return self._p_mean_iid[0] + self._p_mean_iid[1] / t
    
    def estimate_performance_mean_in_conditional_setup(self, t):
        t = np.asarray(t).reshape(-1)
        if self._p_mean_cond is None:
            self._estimate_params_for_mean(iid_data=False)
        return self._p_mean_cond[0] + self._p_mean_cond[1] / t

    def estimate_performance_var_in_iid_setup(self, n, t):
        n = np.asarray(n).reshape(-1)
        t = np.asarray(t).reshape(-1)
        if self.model_v_iid is None:
            self._estimate_params_for_iid_var(n)
        
        nt_pairs = it.product(n, t)
        queries = np.array([
            [np.ones_like(_t) / _n, 1 / (_n * _t), 1 / (_n * _t**2), 1 / (_n * _t**3), 1 / _t, 1 / _t**2, 1 / _t**3]
            for _n, _t in nt_pairs
        ])
        return np.maximum(0, self.model_v_iid.predict(queries).reshape(len(n), len(t)))
    
    def estimate_performance_var_in_conditional_setup(self, t):
        t = np.asarray(t).reshape(-1)
        if self.model_v_cond is None:
            self._estimate_params_for_conditional_var()
        
        return np.maximum(0, self.model_v_cond.predict(np.array([np.ones_like(t), 1 / t, 1 / t**2, 1 / t**3]).T))
