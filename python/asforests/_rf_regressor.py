import numpy as np
from scipy.stats import bootstrap
import sklearn.ensemble
import logging
from ._grower import ForestGrower

class RandomForestRegressor(sklearn.ensemble.RandomForestRegressor):
    
    def __init__(self, step_size = 5, w_min = 50, epsilon = 10, extrapolation_multiplier = 1000, bootstrap_repeats = 5, max_trees = None, stop_when_horizontal = True, random_state = None):
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
        self.args = {
            "step_size": step_size,
            "w_min": w_min,
            "delta": w_min,
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
        
    def get_score_generator(self, X, y):
        
        # stuff to efficiently compute OOB
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
        def get_mse_score(y_pred):
            return np.mean((y_pred - y)**2)
        
        # this is a variable that is being used by the supplier
        self.y_oob = np.zeros(X.shape[0])
        
        def f():
            
            while True: # the generator will add trees forever
                
                # add a new tree
                self.n_estimators += self.step_size
                super(RandomForestRegressor, self).fit(X, y)

                # update distribution based on last trees
                for t in range(self.n_estimators - self.step_size, self.n_estimators):

                    # get i-th last tree
                    last_tree = self.estimators_[t]

                    # get indices not used for training
                    unsampled_indices = get_unsampled_indices(last_tree)

                    # update Y_prob with respect to OOB probs of the tree
                    y_oob_tree = self.predict_tree(t, X[unsampled_indices])

                    # update forest's prediction
                    self.y_oob[unsampled_indices] = (y_oob_tree + t * self.y_oob[unsampled_indices]) / (t + 1) # this will converge according to the law of large numbers

                    yield get_mse_score(self.y_oob)
        
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