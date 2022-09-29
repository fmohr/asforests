import numpy as np
from scipy.stats import bootstrap
import sklearn.ensemble
import logging

class RandomForestClassifier(sklearn.ensemble.RandomForestClassifier):
    
    def __init__(self, validation_fold_size = 0.2, step_size = 5, w_min = 20, epsilon = 0.01, extrapolation_multiplier = 1000, max_trees = None, stop_when_horizontal = True, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='sqrt', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, n_jobs=None, random_state=None, verbose=0, class_weight=None, ccp_alpha=0.0, max_samples=None):
        self.kwargs = {
            "n_estimators": 0, # will be increased steadily
            "criterion": criterion,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "min_weight_fraction_leaf": min_weight_fraction_leaf,
            "max_features": max_features,
            "max_leaf_nodes": max_leaf_nodes,
            "min_impurity_decrease": min_impurity_decrease,
            "bootstrap": bootstrap,
            "oob_score": False,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "verbose": verbose,
            "class_weight": class_weight,
            "ccp_alpha": ccp_alpha,
            "max_samples": max_samples,
            "warm_start": True
        }
        super().__init__(**self.kwargs)
        
        if random_state is None:
            random_state = 0
        if type(random_state) == np.random.RandomState:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
    
        self.stop_when_horizontal = stop_when_horizontal
    
        self.validation_fold_size = validation_fold_size
        self.step_size = step_size
        self.w_min = w_min
        self.epsilon = epsilon
        self.extrapolation_multiplier = extrapolation_multiplier
        self.max_trees = max_trees
        self.logger = logging.getLogger("ASRFClassifier")
        
    def estimate_slope(self, window:np.ndarray):
        window = window[int(len(window) * 0.8):]
        window_domain = np.array(range(len(window)))
        def get_slope(indices = slice(0, len(window))):
            cov = np.cov(np.array([window_domain[indices], window[indices]]))
            slope_in_window = cov[0,1] / cov[0,0]
            return slope_in_window
        
        if min(window) < max(window):
            try:
                result = bootstrap((list(range(len(window))),), get_slope, vectorized = False, n_resamples = 20)
                ci = result.confidence_interval
                return max(np.abs([ci.high, ci.low]))
            except ValueError:
                return 0
        else:
            return 0
        
    def grow(self, info_supplier, d):
        
        # initialize state variables and history
        open_dims = list(range(d))
        self.alpha = [1 for i in open_dims]
        self.histories = [[] for i in open_dims]
        self.start_of_convergence_window = [0 for i in open_dims]
        self.is_cauchy = [False for i in open_dims]
        self.diffs = [None for i in open_dims]
        
        self.slope_hist = []
        self.cauchy_history = []
        converged = False
        
        # now start training
        t = 1
        while d > 0 and (self.max_trees is None or t <= self.max_trees):
            infos = info_supplier(self.alpha)
            
            for i in open_dims.copy():

                # update history for this dimension
                history = self.histories[i]
                last_info = infos[i]
                history.append(last_info)

                if not converged:
                    
                    # update delta in the necessary region
                    cur_window_start = self.start_of_convergence_window[i]
                    if self.diffs[i] is None:
                        self.diffs[i] = np.array([[0]])
                    else:
                        self.diffs[i] = np.column_stack([self.diffs[i], np.nan * np.zeros(self.diffs[i].shape[0])])
                        self.diffs[i] = np.row_stack([self.diffs[i], np.nan * np.zeros(self.diffs[i].shape[1])])
                    diffs = self.diffs[i]
                    for j, vj in enumerate(history[cur_window_start:], start = cur_window_start):
                        diffs[j,-1] = diffs[-1,j] = np.linalg.norm(vj - last_info)


                    # determine whether and up to which point in the past the Cauchy criterion is still valid
                    violations_of_new_point_to_window = diffs[-1,cur_window_start:-1] > self.epsilon
                    violating_indices = np.where(violations_of_new_point_to_window)[0]
                    last_violation = None if len(violating_indices) == 0 else violating_indices[-1]
                    #print(t, cur_window_start, last_violation, diffs)
                    if last_violation is None:
                        if not self.is_cauchy[i] and len(violations_of_new_point_to_window) * self.step_size >= self.w_min:
                            self.is_cauchy[i] = True
                    else: # if the sequence was not convergent before and we still observe some violation in the window, adjust the window start
                        self.start_of_convergence_window[i] += last_violation + 1
                        cur_window_start = self.start_of_convergence_window[i]
                        if (t - cur_window_start) * self.step_size < self.w_min:
                            self.logger.debug(f"Resetting convergence at t = {t}. Last violation was at {last_violation}. Current window only spans {(t - cur_window_start) * self.step_size} trees")
                            self.is_cauchy[i] = False

                # if the dimension is Cauchy convergent, also estimate the slope
                if self.is_cauchy[i]:
                    window = np.array(history[cur_window_start:])
                    slope = self.estimate_slope(window) / self.step_size
                    self.slope_hist.append(slope)
                    if np.abs(slope * self.extrapolation_multiplier) < self.epsilon:
                        converged = True
                        if self.stop_when_horizontal:
                            open_dims.remove(i)
                            self.alpha[i] = 0
                            d -= 1
                else:
                    self.slope_hist.append(np.nan)
                self.cauchy_history.append(self.is_cauchy.copy())
            t += 1
               
    def fit(self, X, y):
        
        # set numbers of trees to 0
        self.warm_start = False
        self.estimators_ = []
        self.n_estimators = 0
        self.warm_start = True
        
        # memorize labels
        labels = list(np.unique(y))
        
        # set validation data apart, and replace the label vector by a binary matrix 
        X_train, X_valid, y_train, y_valid = sklearn.model_selection.train_test_split(X, y, random_state = self.random_state, test_size = self.validation_fold_size)
        n, k = len(y_valid), len(labels)
        Y_valid = np.zeros((n, k))
        for i, true_label in enumerate(y_valid):
            Y_valid[i,labels.index(true_label)] = 1
        
        # create a function that can efficiently compute the log-loss for a probability distribution
        mask_for_log_loss = np.where(Y_valid == 1)
        def get_log_loss(Y_prob):
            return -np.mean(np.log(np.clip(Y_prob[mask_for_log_loss], self.beta, 1 - self.beta)))

        def get_brier_score(Y_prob):
            return np.mean(np.sum((Y_prob - Y_valid)**2, axis=1))
        
        # this is a variable that is being used by the supplier
        self.y_prob_valid = np.zeros((X_valid.shape[0], len(labels)))
        
        def supplier_for_all_probas(alpha): # alpha not used here
            
            # add a new tree
            self.rf.n_estimators += self.step_size
            self.rf.fit(X_train, y_train)
            t = self.rf.n_estimators

            # get last tree
            last_tree = self.rf.estimators_[-1]
            y_prob_valid_tree = last_tree.predict_proba(X_valid)
            
            # update forest's prediction
            y_prob_valid_new = (y_prob_valid_tree + (t-1) * self.y_prob_valid) / t # this will converge according to the law of large numbers

            # compute changes in distributions
            Y_prob_diff_valid = self.y_prob_valid - y_prob_valid_new
            self.y_prob_valid = y_prob_valid_new
            
            # compute performance diffs of relevant other anchors to new one
            return self.y_prob_valid.ravel()
        
        def supplier_for_highest_proba(alpha): # alpha not used here
            
            # add a new tree
            self.rf.n_estimators += self.step_size
            self.rf.fit(X_train, y_train)
            t = self.rf.n_estimators

            # get last tree
            last_tree = self.rf.estimators_[-1]
            y_prob_valid_tree = last_tree.predict_proba(X_valid)
            
            # update forest's prediction
            y_prob_valid_new = (y_prob_valid_tree + (t-1) * self.y_prob_valid) / t # this will converge according to the law of large numbers

            # compute changes in distributions
            Y_prob_diff_valid = self.y_prob_valid - y_prob_valid_new
            self.y_prob_valid = y_prob_valid_new
            
            mask = np.argmax(self.y_prob_valid, axis = 1)
            
            # compute performance diffs of relevant other anchors to new one
            return [r[e] for r, e in zip(self.y_prob_valid, mask)]
        
        def supplier_for_ll(alpha): # alpha not used here
            
            # add a new tree
            self.rf.n_estimators += self.step_size
            self.rf.fit(X_train, y_train)
            t = self.rf.n_estimators

            # get last tree
            last_tree = self.rf.estimators_[-1]
            y_prob_valid_tree = last_tree.predict_proba(X_valid)
            
            # update forest's prediction
            y_prob_valid_new = (y_prob_valid_tree + (t-1) * self.y_prob_valid) / t # this will converge according to the law of large numbers

            # compute changes in distributions
            Y_prob_diff_valid = self.y_prob_valid - y_prob_valid_new
            self.y_prob_valid = y_prob_valid_new
            
            # compute performance diffs of relevant other anchors to new one
            return [get_log_loss(self.y_prob_valid)]
        
        def supplier_for_bs(alpha): # alpha not used here
            
            # add a new tree
            self.n_estimators += self.step_size
            super(RandomForestClassifier, self).fit(X_train, y_train)
            t = self.n_estimators

            # get last tree
            last_tree = self.estimators_[-1]
            y_prob_valid_tree = last_tree.predict_proba(X_valid)
            
            # update forest's prediction
            y_prob_valid_new = (y_prob_valid_tree + (t-1) * self.y_prob_valid) / t # this will converge according to the law of large numbers

            # compute changes in distributions
            Y_prob_diff_valid = self.y_prob_valid - y_prob_valid_new
            self.y_prob_valid = y_prob_valid_new
            
            # compute performance diffs of relevant other anchors to new one
            return [get_brier_score(self.y_prob_valid)]
        
        # always use Brier score supplier
        self.grow(supplier_for_bs, 1)