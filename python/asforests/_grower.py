import numpy as np
from scipy.stats import bootstrap
import logging

class ForestGrower:
    
    def __init__(self, info_supplier, d, step_size, w_min, epsilon, extrapolation_multiplier, max_trees, random_state, stop_when_horizontal, logger):
        
        if w_min <= step_size:
            raise ValueError("\"w_min\" must be strictly bigger than the value of \"step_size\".")
        
        self.info_supplier = info_supplier
        self.d = d
        self.step_size = step_size
        self.w_min = w_min
        self.epsilon = epsilon
        self.extrapolation_multiplier = extrapolation_multiplier
        self.max_trees = max_trees
        self.logger = logger
        self.random_state = random_state
        self.stop_when_horizontal = stop_when_horizontal
        
    def estimate_slope(self, window:np.ndarray):
        max_index_to_have_two_values = max(0, len(window) - self.w_min)
        last_20_percent_index = int(len(window) * 0.8)
        window = window[min(max_index_to_have_two_values, last_20_percent_index):]
        if len(window) == 1:
            raise ValueError(f"Window must have length of more than 1.")
        window_domain = np.array(range(0, len(window) * self.step_size + 1, self.step_size))
        self.logger.debug(f"\tEstimating slope for window of size {len(window)}.")
        def get_slope(indices = slice(0, len(window))):
            if len(np.unique(indices)) == 1: # if there is just one value in the bootstrap sample, return nan
                return np.nan
            cov = np.cov(np.array([window_domain[indices], window[indices]]))
            slope_in_window = cov[0,1] / cov[0,0]
            return slope_in_window
        
        if min(window) < max(window):
            try:
                result = bootstrap((list(range(len(window))),), get_slope, vectorized = False, n_resamples = 2, random_state = self.random_state, method = "percentile")
                ci = result.confidence_interval
                return max(np.abs([ci.high, ci.low]))
            except ValueError:
                return 0
        else:
            return 0
    
    def grow(self):
        
        self.reset()
        
        # now start training
        self.logger.info(f"Start training with following parameters:\n\tStep Size: {self.step_size}\n\tepsilon: {self.epsilon}")
        while self.d > 0 and (self.max_trees is None or self.t * self.step_size <= self.max_trees):
            self.step()
        self.logger.info("Forest grown completely. Stopping routine!")
        
    def reset(self):
        
        # initialize state variables and history
        self.open_dims = open_dims = list(range(self.d))
        self.histories = [[] for i in open_dims]
        self.start_of_convergence_window = [0 for i in open_dims]
        self.is_cauchy = [False for i in open_dims]
        self.diffs = [None for i in open_dims]
        
        self.slope_hist = []
        self.cauchy_history = []
        self.converged = False
        self.t = 1
        
    def step(self):
        
        self.logger.debug(f"Starting Iteration {self.t}.")
        self.logger.debug(f"\tAdding {self.step_size} trees to the forest.")
        score = next(self.info_supplier)
        self.logger.debug(f"\tDone. Forest size is now {self.t * self.step_size}. Score: {score}")

        for i in self.open_dims.copy():

            self.logger.debug(f"\tChecking dimension {i}. Cauchy criterion in this dimension: {self.is_cauchy[i]}")

            # update history for this dimension
            history = self.histories[i]
            last_info = score
            history.append(last_info)

            if not self.converged:

                # update delta in the necessary region
                cur_window_start = self.start_of_convergence_window[i]
                self.logger.debug(f"\tForest not converged in criterion  {i}. Computing differences from forest size {cur_window_start * self.step_size} on.")
                if self.diffs[i] is None:
                    self.diffs[i] = np.array([[0]])
                else:
                    self.diffs[i] = np.column_stack([self.diffs[i], np.nan * np.zeros(self.diffs[i].shape[0])])
                    self.diffs[i] = np.row_stack([self.diffs[i], np.nan * np.zeros(self.diffs[i].shape[1])])
                diffs = self.diffs[i]
                for j, vj in enumerate(history[cur_window_start:], start = cur_window_start):
                    diffs[j,-1] = diffs[-1,j] = np.linalg.norm(vj - last_info)
                diffs_of_new_point_in_current_window = diffs[-1,cur_window_start:-1]
                self.logger.debug(f"\tDone. Computed {len(history[cur_window_start:])} differences. Max diff in window is: {np.round(max(diffs_of_new_point_in_current_window), 3) if len(diffs_of_new_point_in_current_window) > 0 else 0}")

                # determine whether and up to which point in the past the Cauchy criterion is still valid
                violations_of_new_point_to_window = diffs_of_new_point_in_current_window > self.epsilon
                violating_indices = np.where(violations_of_new_point_to_window)[0]
                last_violation = None if len(violating_indices) == 0 else violating_indices[-1]
                self.logger.debug(f"\tLast violation within the analyzed window: {last_violation}")
                if last_violation is None:
                    if not self.is_cauchy[i] and len(violations_of_new_point_to_window) * self.step_size >= self.w_min:
                        self.logger.info(f"\tVerified Cauchy criterion for dimension {i}")
                        self.is_cauchy[i] = True
                else: # if the sequence was not convergent before and we still observe some violation in the window, adjust the window start
                    self.start_of_convergence_window[i] += last_violation + 1
                    cur_window_start = self.start_of_convergence_window[i]
                    self.logger.debug(f"\tSet start of convergence window to {self.start_of_convergence_window[i]}. Window now spans {(self.t - cur_window_start) * self.step_size} trees")
                    if (self.t - cur_window_start) * self.step_size < self.w_min:
                        self.logger.debug(f"\tResetting convergence at t = {self.t}. Last violation was at {last_violation}. Current window only spans {(self.t - cur_window_start) * self.step_size} trees")
                        self.is_cauchy[i] = False

            # if the dimension is Cauchy convergent, also estimate the slope
            if self.is_cauchy[i]:
                window = np.array(history[cur_window_start:])
                self.logger.debug(f"\tCauchy holds. Checking slope in window of length {len(window)} with entries since iteration {cur_window_start}.")
                if len(window) <= 1:
                    self.logger.info("RUNNING BEFORE EXCEPTION")
                    raise ValueError(f"Detected Cauchy criterion in a window of length {len(window)}, but such a window must have length at least 2.")
                    self.logger.info("RUNNING AFTER EXCEPTION")
                    raise Exception()

                slope = self.estimate_slope(window)
                self.logger.debug(f"\tEstimated slope is {slope}. Maximum deviation on {self.extrapolation_multiplier} trees is {np.round(slope * self.extrapolation_multiplier, 4)}.")
                self.slope_hist.append(slope)
                if np.abs(slope * self.extrapolation_multiplier) < self.epsilon:
                    self.logger.info(f"\tDetected convergence (Cauchy + horizontal).")
                    self.converged = True
                    if self.stop_when_horizontal:
                        self.open_dims.remove(i)
                        self.d -= 1
            else:
                self.slope_hist.append(np.nan)
            self.cauchy_history.append(self.is_cauchy.copy())
        self.t += 1