import numpy as np
from scipy.stats import bootstrap
import logging
import time

def get_dummy_info_supplier(history):
    
    def iterator():
        for e in history:
            yield e
    return iterator()

class ForestGrower:
    
    def __init__(self, info_supplier, d, step_size, w_min, epsilon, delta, extrapolation_multiplier, max_trees, random_state, stop_when_horizontal, bootstrap_repeats, logger = None):
        
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
        self.random_state = np.random.RandomState(random_state) if type(random_state) == int else random_state
        self.stop_when_horizontal = stop_when_horizontal
        self.bootstrap_repeats = bootstrap_repeats
        self.bootstrap_init_seed = self.random_state.randint(10**5)
        self.cov_times = []
        
    def estimate_slope(self, window:np.ndarray):
        window = window[-self.delta:]
        if len(window) < 2:
            raise ValueError(f"Window must have length of at least 2.")
        window_domain = np.arange(0, len(window))
        self.logger.debug(f"\tEstimating slope for window of size {len(window)}.")
        
        def get_slope(indices = slice(0, len(window))):
            self.logger.debug(indices)
            cov = np.cov(window_domain[indices], window[indices])
            return cov[0,1] / cov[0,0] if cov[0,0] > 0 else np.nan # if there is just one value in the bootstrap sample, return nan
        
        if np.min(window) < np.max(window):
            try:
                if self.bootstrap_repeats > 1:
                    start = time.time()
                    result = bootstrap((list(range(len(window))),), get_slope, vectorized = False, n_resamples = self.bootstrap_repeats, random_state = self.bootstrap_init_seed + self.t, method = "percentile")
                    self.cov_times.append(time.time() - start)
                    ci = result.confidence_interval
                    return np.max(np.abs([ci.high, ci.low]))
                else:
                    dummy = list(range(len(window)))
                    return np.abs(get_slope(dummy))
                    
            except ValueError:
                return 0
        else:
            return 0
    
    def grow(self):
        
        self.reset()
        
        # now start training
        self.logger.info(f"Start training with following parameters:\n\tStep Size: {self.step_size}\n\tepsilon: {self.epsilon}")
        while self.d > 0 and (self.max_trees is None or len(self.histories[0]) + self.step_size <= self.max_trees):
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
        
        self.time_stats_steps = []
        self.time_stats_supplier = []
        self.time_stats_preamble = []
        self.time_stats_cauchy = []
        self.time_stats_slope = []
        
    def step(self):
        start_step = time.time()
        self.logger.debug(f"Starting Iteration {self.t}.")
        self.logger.debug(f"\tAdding {self.step_size} trees to the forest.")
        new_scores = []
        for i in range(self.step_size):
            start = time.time()
            new_scores.append(next(self.info_supplier))
            self.time_stats_supplier.append(time.time() - start)
        self.logger.debug(f"\tDone. Forest size is now {self.t * self.step_size}. Scores to be added: {new_scores}")

        for i in self.open_dims.copy():

            self.logger.debug(f"\tChecking dimension {i}. Cauchy criterion in this dimension: {self.is_cauchy[i]}")

            # update history for this dimension
            start = time.time()
            cur_window_start = self.start_of_convergence_window[i]
            history = self.histories[i]
            history.extend(new_scores)
            history = np.fromiter(history, dtype=np.float64) # convert locally to numpy for faster processing. Roughly twice as fast as np.array
            s_min = self.s_mins[i]
            s_max = self.s_maxs[i]
            self.time_stats_preamble.append(time.time() - start)
            
            if not self.converged:

                # criterion 1: current Cauchy window size (here given by index of where the convergence window starts)
                self.logger.debug(f"\tForest not converged in criterion  {i}. Computing differences from forest size {cur_window_start * self.step_size} on.")
                start = time.time()
                adjust_min = False
                adjust_max = False
                for score in history[-self.step_size:]:
                    if s_min > score:
                        s_min = self.s_mins[i] = score
                        adjust_min = True
                    if score > s_max:
                        s_max = self.s_maxs[i] = score
                        adjust_max = True
                
                # if the window must be adjusted
                if s_max > s_min + self.epsilon:
                    if adjust_min:
                        violations = history[cur_window_start:] > (s_min + self.epsilon)
                        if np.count_nonzero(violations) > 0:
                            cur_window_start = np.where(violations)[0][-1] + 1 + cur_window_start
                        s_max = self.s_maxs[i] = np.max(history[cur_window_start:]) if cur_window_start < len(history) else -np.inf
                        
                    elif adjust_max:
                        violations = history[cur_window_start:] < (s_max - self.epsilon)
                        if np.count_nonzero(violations) > 0:
                            cur_window_start = np.where(violations)[0][-1] + 1 + cur_window_start
                        s_min = self.s_mins[i] = np.min(history[cur_window_start:]) if cur_window_start < len(history) else np.inf
                    else:
                        raise Exception("There must be an adjustment!")
                self.start_of_convergence_window[i] = cur_window_start
                w = len(history) - cur_window_start
                self.is_cauchy[i] = w >= self.w_min
                self.time_stats_cauchy.append(time.time() - start)

            # if the dimension is Cauchy convergent, also estimate the slope
            if self.is_cauchy[i]:
                start = time.time()
                window = history[cur_window_start:]
                self.logger.debug(f"\tCauchy holds. Checking slope in window of length {len(window)} with entries since iteration {cur_window_start}.")
                if len(window) <= 1:
                    raise ValueError(f"Detected Cauchy criterion in a window of length {len(window)}, but such a window must have length at least 2.")
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
                self.time_stats_slope.append(time.time() - start)
            else:
                self.slope_hist.append(np.nan)
            self.cauchy_history.append(self.is_cauchy.copy())
        self.t += 1
        self.time_stats_steps.append(time.time() - start_step)