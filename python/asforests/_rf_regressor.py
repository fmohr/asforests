import time
import numpy as np
from scipy.stats import bootstrap
import sklearn.ensemble
import logging
from ._grower import ForestGrower

class RandomForestRegressor(sklearn.ensemble.RandomForestRegressor):
    
    def __init__(self, step_size = 5, w_min = 50, delta = 10, epsilon = 10, extrapolation_multiplier = 1000, bootstrap_repeats = 5, max_trees = None, stop_when_horizontal = True, random_state = None, prediction_map_for_scoring = lambda x: (x - np.median(y_train)) / (np.max(y_train) - np.min(y_train))):
        self.kwargs = {
            "n_estimators": 0, # will be increased steadily
            "oob_score": False,
            "warm_start": True
        }
        super().__init__(**self.kwargs)
        
        if random_state is None:
            random_state = 0
        if type(random_state) == np.random.RandomState:
            self.random_state = random_state
        else:
            self.random_state = np.random.RandomState(random_state)
   
        self.step_size = step_size
        self.w_min = w_min
        self.epsilon = epsilon
        self.extrapolation_multiplier = extrapolation_multiplier
        self.max_trees = max_trees
        self.bootstrap_repeats = bootstrap_repeats
        self.stop_when_horizontal = stop_when_horizontal
        self.prediction_map_for_scoring = prediction_map_for_scoring
        self.args = {
            "step_size": step_size,
            "w_min": w_min,
            "delta": delta,
            "epsilon": epsilon,
            "extrapolation_multiplier": extrapolation_multiplier,
            "bootstrap_repeats": bootstrap_repeats,
            "max_trees": max_trees,
            "stop_when_horizontal": stop_when_horizontal
        }
        self.logger = logging.getLogger("ASRFRegressor")
        
    def __str__(self):
        return "ASRFRegressor"
        
    def predict_tree(self, tree_id, X):
        return self.estimators_[tree_id].predict(X)
        
    def get_score_generator(self, X, y, validation_size = 0.0, random_state = None):
        
        # check whether performance is checked based on OOB or a separate validation fold
        oob = validation_size == 0
        if not oob:
            X_train, X_val, y_train, y_val = sklearn.model_selection.train_test_split(X, y, test_size = validation_size, random_state = random_state)
        else:
            X_train, X_val, y_train, y_val = X, X, y, y
        
        # stuff to efficiently compute OOB
        if oob:
            n_samples = y.shape[0]
            n_samples_bootstrap = sklearn.ensemble._forest._get_n_samples_bootstrap(
                n_samples,
                self.max_samples,
            )
            def get_unsampled_indices(tree):
                return sklearn.ensemble._forest._generate_unsampled_indices(
                    tree.random_state,
                    n_samples,
                    n_samples_bootstrap,
                )
        
        # create a function that can efficiently compute the MSE
        y_ref = y if self.prediction_map_for_scoring is None else self.prediction_map_for_scoring(y)
        def get_mse_score(y_pred):
            if self.prediction_map_for_scoring is not None:
                y_pred = self.prediction_map_for_scoring(y_pred)
            return np.mean((y_pred - y_ref)**2)
        
        # this is a variable that is being used by the supplier
        self.y_pred = np.zeros(X.shape[0])
        self.all_indices = list(range(len(y)))
        
        def f():
            
            while True: # the generator will add trees forever
                
                # add a new tree
                start = time.time()
                self.n_estimators += self.step_size
                super(RandomForestRegressor, self).fit(X, y)
                traintime = time.time() - start

                # update distribution based on last trees
                for t in range(self.n_estimators - self.step_size, self.n_estimators):
                    
                    start = time.time()

                    # get i-th last tree
                    last_tree = self.estimators_[t]

                    # get prediction of tree on the indices relevant for it
                    relevant_indices = get_unsampled_indices(last_tree) if oob else self.all_indices # this is what J is in the paper
                    y_pred_tree = self.predict_tree(t, X[relevant_indices])

                    # update forest's prediction
                    self.y_pred[relevant_indices] = (y_pred_tree + t * self.y_pred[relevant_indices]) / (t + 1) # this will converge according to the law of large numbers

                    pred_time = time.time() - start
                    start = time.time()
                    score = get_mse_score(self.y_pred)
                    score_time = time.time() - start
                    yield score, traintime / self.step_size, pred_time, score_time
        
        return f() # creates the generator and returns it
               
    def reset(self):
        # set numbers of trees to 0
        self.warm_start = False
        self.estimators_ = []
        self.n_estimators = 0
        self.warm_start = True
    
    def fit(self, X, y):
        self.reset()
        gen = self.get_score_generator(X, y)
        grower = ForestGrower(gen,  d = 1, logger = self.logger, random_state = self.random_state, **self.args)
        grower.grow()
        self.histories = grower.histories