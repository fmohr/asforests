import numpy as np
from scipy.stats import bootstrap
import logging

def get_dummy_info_supplier(history):
    
    def iterator():
        for e in history:
            yield e
    return iterator()

class ForestGrower:
    
    def __init__(self, info_supplier, d, step_size, w_min, epsilon, delta, extrapolation_multiplier, max_trees, random_state, stop_when_horizontal, bootstrap_repeats, logger):
        
        if w_min <= step_size:
            raise ValueError("\"w_min\" must be strictly bigger than the value of \"step_size\".")
        
        self.info_supplier = info_supplier
        self.d = d
        self.step_size = step_size
        self.w_min = w_min
        self.delta = delta
        self.epsilon = epsilon
        self.extrapolation_multiplier = extrapolation_multiplier
        self.max_trees = max_trees
        self.logger = logger
        self.random_state = random_state
        self.stop_when_horizontal = stop_when_horizontal
        self.bootstrap_repeats = bootstrap_repeats
        
    def estimate_slope(self, window:np.ndarray):
        max_index_to_have_two_values = max(0, len(window) - self.w_min)
        window = window[min(max_index_to_have_two_values, len(window) - self.delta):]
        if len(window) == 1:
            raise ValueError(f"Window must have length of more than 1.")
        window_domain = np.array(range(0, len(window) * self.step_size + 1, self.step_size))
        self.logger.debug(f"\tEstimating slope for window of size {len(window)}.")
        def get_slope(indices = slice(0, len(window))):
            self.logger.debug(indices)
            if len(np.unique(indices)) == 1: # if there is just one value in the bootstrap sample, return nan
                return np.nan
            cov = np.cov(np.array([window_domain[indices], window[indices]]))
            slope_in_window = cov[0,1] / cov[0,0]
            return slope_in_window
        
        if min(window) < max(window):
            try:
                if self.bootstrap_repeats > 1:
                    result = bootstrap((list(range(len(window))),), get_slope, vectorized = False, n_resamples = self.bootstrap_repeats, random_state = self.random_state + self.t, method = "percentile")
                    ci = result.confidence_interval
                    return max(np.abs([ci.high, ci.low]))
                else:
                    return np.abs(get_slope(list(range(len(window)))))
                    
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
        self.s_mins = [np.inf for i in open_dims]
        self.s_maxs = [-np.inf for i in open_dims]
        
        self.slope_hist = []
        self.cauchy_history = []
        self.converged = False
        self.t = 1
        
        window_domain = range(1, self.delta + 1)
        mean_domain = np.mean(window_domain)
        self.qa = [i - mean_domain for i in window_domain]
        
    def step(self):
        self.logger.debug(f"Starting Iteration {self.t}.")
        self.logger.debug(f"\tAdding {self.step_size} trees to the forest.")
        score = next(self.info_supplier)
        self.logger.debug(f"\tDone. Forest size is now {self.t * self.step_size}. Score: {score}")

        for i in self.open_dims.copy():

            self.logger.debug(f"\tChecking dimension {i}. Cauchy criterion in this dimension: {self.is_cauchy[i]}")

            # update history for this dimension
            cur_window_start = self.start_of_convergence_window[i]
            history = self.histories[i]
            last_info = score
            s_min = self.s_mins[i]
            s_max = self.s_maxs[i]
            history.append(last_info)
            
            if not self.converged:

                # criterion 1: current Cauchy window size (here given by index of where the convergence window starts)
                self.logger.debug(f"\tForest not converged in criterion  {i}. Computing differences from forest size {cur_window_start * self.step_size} on.")
                if s_min > last_info:
                    s_min = self.s_mins[i] = last_info
                if last_info > s_max:
                    s_max = self.s_maxs[i] = last_info
                
                if s_max - s_min > self.epsilon:
                    cur_window_start = len(history)
                    s_min_tmp = s_max_tmp = last_info
                    while s_max_tmp - s_min_tmp <= self.epsilon:
                        s_max = s_max_tmp
                        s_min = s_min_tmp
                        cur_window_start -= 1
                        s_min_tmp = min(s_min, history[cur_window_start - 1])
                        s_max_tmp = max(s_max, history[cur_window_start - 1])
                    self.start_of_convergence_window[i] = cur_window_start
                w = len(history) - cur_window_start
                self.is_cauchy[i] = w >= self.w_min

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
                if np.isnan(slope):
                    slope = (max(window) - min(window)) / (len(window) - 1)
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