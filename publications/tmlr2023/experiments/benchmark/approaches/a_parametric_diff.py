import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy.optimize import nnls

from .approach import Approach

import logging
from tqdm import tqdm

from sklearn.linear_model import Ridge


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
        self.model_v_cond = None

    
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
        self.model_v_cond = None
        self.logger.info(f"Added prediction matrix #{len(self.prediction_matrices)} of shape {prediction_matrix.shape}")
    
    
    def _get_schedule_for_max_anchor(self, max_anchor=10**3):
        """
           :param max_anchor: this variable define up to which t we want to create data points. Could be `b` or some constant
        """
        schedule = sorted(set([int(np.round(2**(i / 4))) for i in range(int(4 * np.log2(max_anchor) + 1))]))
        #return np.round(np.concatenate([
         #   np.linspace(1, 10**2, 10)
            #np.linspace(10**2, max_anchor, 10)
        #])).astype(int)
        if max_anchor not in schedule:
            schedule.append(max_anchor)
        return np.array(schedule)

    def _sample_ensemble_performances_at_schedule(self):
        b = len(self.prediction_matrices)
        max_anchor = max(10**3, b)
        self.logger.info(f"Drawing {self.num_simulated_ensembles} ensembles of size {b}")
        schedule = self._get_schedule_for_max_anchor(max_anchor)

        # compute data for parametric learning problem
        sizes = []
        errors = []
        if self.anchors == "full":
            self.logger.info(f"Computing database with full schedule")
            ensembles_in_batch = self.random_state.randint(0, b, size=(max_anchor, self.num_simulated_ensembles))
            for ensemble in ensembles_in_batch:
                ensemble_prediction_matrices = np.zeros(self.prediction_matrices[0].shape)
                for s, i in enumerate(ensemble, start=1):
                    ensemble_prediction_matrices += (self.prediction_matrices[i] - ensemble_prediction_matrices) / s
                    error_of_this_ensemble = ((ensemble_prediction_matrices - self.y_oh)**2).mean(axis=0).sum()
                    sizes.append(s)
                    errors.append(error_of_this_ensemble)
            errors = np.array(errors)
            
        elif self.anchors.startswith("power"):
            self.logger.info(f"Computing database. Using geometric schedule with anchors {schedule}.")
            matrices = np.array(self.prediction_matrices)
            for size in schedule: # create data for each anchor on the schedule. The anchor is used for both ensemble size and number of application instances (relevant for estimate of V[Z_nt])
                required_entries = size * self.num_simulated_ensembles * matrices.shape[1] * matrices.shape[2]
                MAX_ENTRIES_IN_MATRIX = 10**8
                max_batch_size = int(np.ceil(MAX_ENTRIES_IN_MATRIX / (size * matrices.shape[1] * matrices.shape[2])))
                num_batches = int(np.ceil(required_entries / MAX_ENTRIES_IN_MATRIX))
                ensemble_prediction_matrix_collection = []
                self.logger.debug(f"Computing {self.num_simulated_ensembles} ensemble prediction matrices at anchor {size} in {num_batches} batches.")
                num_remaining_entries = self.num_simulated_ensembles
                for batch_idx in range(num_batches):
                    actual_batch_size = min(num_remaining_entries, max_batch_size)
                    ensembles_in_batch = self.random_state.randint(0, b, size=(max_anchor, actual_batch_size))
                    ensemble_member_predictions = matrices[ensembles_in_batch[:size].ravel()].reshape((size, ensembles_in_batch.shape[1]) + matrices.shape[1:])
                    ensemble_prediction_matrix_collection.append(ensemble_member_predictions.mean(axis=0))
                    num_remaining_entries -= actual_batch_size
                ensemble_prediction_matrices = np.concatenate(ensemble_prediction_matrix_collection)
                self.logger.debug(f"Computed ensemble prediction matrices at anchor {size} of shape {ensemble_prediction_matrices.shape}.")
                errors_for_this_size = ((ensemble_prediction_matrices - self.y_oh)**2).mean(axis=1).sum(axis=1)
                added_errors = errors_for_this_size.ravel()
                sizes.extend(len(added_errors) * [size])
                errors.extend(added_errors)
        self._sample_of_ensemble_performances = pd.DataFrame(data={"t": sizes, "Z_nt": errors})
        self.logger.info(f"Done. Database has {len(self._sample_of_ensemble_performances)} entries.")

    
    def _estimate_params_for_mean(self):
        # if we do not have enough observations, return 0
        # TODO: return empirical mean
        b = len(self.prediction_matrices)
        
        # create permutations
        self._sample_ensemble_performances_at_schedule()

        # estimate parameters for mean
        self.logger.info(f"Now fitting the model.")
        X = np.column_stack((np.ones(len(self._sample_of_ensemble_performances)), 1 / np.array(self._sample_of_ensemble_performances["t"])))
        self._p_mean = lstsq(X, self._sample_of_ensemble_performances["Z_nt"])[0]


        #import matplotlib.pyplot as plt
        #fig, ax = plt.subplots()
        #domain = np.linspace(0, 1000, 100)
        #ax.plot(domain, self._p_mean[0] + self._p_mean[1] / domain, marker="o")
        #plt.show()
        
        self.logger.info(f"Done, stored values {self._p_mean}.")
    
    def _estimate_params_for_conditional_var(self):
        # if we do not have enough observations, return 0
        # TODO: return empirical mean
        b = len(self.prediction_matrices)
        if b < 1:
            self._p_cvar = np.zeros(4)
            return

        # sample new ensembles 
        # OBSOLETE, BECAUSE THIS IS ALSO DONE INTERNALLY WHEN GETTING THE MUS
        #self._sample_ensemble_performances_at_schedule()
        
        #mse_hist = []
        #anchors = [2**i for i in range(14)]
        #for anchor in anchors:

            #sub_frame = self._sample_of_ensemble_performances.sample(n=anchor)
        
        # 
        schedule = self._get_schedule_for_max_anchor(max(10**3, b))
        mus = self.estimate_performance_mean_in_conditional_setup(t=schedule)

        # the ensemble performances at the schedule were implicitly computed, so we can now use them to define the targets
        schedule = []
        sizes = []
        targets = []
        for (t, df_t), mu in zip(self._sample_of_ensemble_performances.groupby("t"), mus):
            sizes.extend([t] * len(df_t))
            targets.extend((mu - df_t["Z_nt"])**2)
            schedule.append(t)
        sizes = np.array(sizes)
        targets = np.array(targets)

        # estimate parameters for mean
        self.logger.info(f"Now fitting a model with {len(targets)} data points.")
        X = np.column_stack((np.ones_like(sizes), 1 / np.array(sizes), 1 / np.array(sizes)**2, 1 / np.array(sizes)**3))
        self.model_v_cond = Ridge(alpha=0.0)
        self.model_v_cond.fit(X, targets, sample_weight=sizes)#1 / (targets**2 + 10**-10))
        #self._p_cvar = lstsq(X, targets)[0]

        if False and len(self.prediction_matrices) > 2:
            #pred = self.model_v_cond.predict(X)
            #pred = (self._p_cvar[0] + self._p_cvar[1] / sizes + self._p_cvar[2] / sizes**2 + self._p_cvar[3] / sizes**3)
            pred = self.estimate_performance_var_in_conditional_setup(sizes)
            mse = ((targets - pred)**2).mean()
            #mse_hist.append(mse)

            # show learning results
            import matplotlib.pyplot as plt
            fig, axs = plt.subplots(1, 3, figsize=(10, 3))
            
            ax = axs[0]
            ax.scatter(targets, pred, c=np.log10(sizes), cmap="Blues")
            ax.grid()
            ax.plot([0, 1], [0, 1], color="black", linestyle="--")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlim([10**-10, 1])
            ax.set_ylim([10**-10, 1])
            ax.set_title(f"Round {len(self.prediction_matrices)} - MSE: {mse}")
            
            ax = axs[1]
            #ax.plot(schedule, self.model_v_cond.predict(np.array([np.ones_like(schedule), 1 / schedule, 1 / schedule **2, 1 / schedule**3]).T))
            plot_schedule = np.arange(1, 1001)
            ax.plot(plot_schedule, self.estimate_performance_var_in_conditional_setup(t=plot_schedule))
            ax.plot(schedule, [
                np.mean(targets[sizes == t])
                for t in np.unique(sizes)
            ])
            ax.grid()
            ax.set_xscale("log")
            ax.set_yscale("log")

            ax = axs[2]
            cols = []
            t_domain = sorted(np.unique(sizes))
            for i, t in enumerate(t_domain):
                mask = sizes == t
                cols.append(targets[mask])
                ax.scatter([i + 1], [pred[mask].mean()], color="red", s=50)
            ax.boxplot(cols)
            ax.set_yscale("log")
            ax.set_xticklabels(t_domain)

            plt.show()
            #exit(0)
        

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
        print(n)
        sizes = np.array(sizes)

        # estimate parameters for mean
        self.logger.info(f"Now fitting {len(targets)} models, one per target.")
        X = np.column_stack((np.ones_like(sizes) / n, 1 / (n * sizes), 1 / (n * sizes**2), 1 / (n * sizes**3), 1 / sizes, 1 / sizes**2, 1 / sizes**3))
        self._p_iidvar = nnls(X, targets)[0]
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
        #if self._p_cvar is None:
            #self._estimate_params_for_conditional_var()
        #return (self._p_cvar[0] + self._p_cvar[1] / t + self._p_cvar[2] / t**2 + self._p_cvar[3] / t**3)
        if self.model_v_cond is None:
            self._estimate_params_for_conditional_var()
        
        return self.model_v_cond.predict(np.array([np.ones_like(t), 1 / t, 1 / t**2, 1 / t**3]).T)
        #else:
            #return np.zeros_like(t)
        
