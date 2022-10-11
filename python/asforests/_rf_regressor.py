import numpy as np
from scipy.stats import bootstrap
import sklearn.ensemble
import logging
from ._grower import ForestGrower

class RandomForestRegressor(sklearn.ensemble.RandomForestRegressor):
    
    def __init__(self, step_size = 5, w_min = 50, epsilon = 10, extrapolation_multiplier = 1000, max_trees = None, stop_when_horizontal = True, random_state = None):
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
        self.args = {
            "step_size": step_size,
            "w_min": w_min,
            "epsilon": epsilon,
            "extrapolation_multiplier": extrapolation_multiplier,
            "max_trees": max_trees,
            "stop_when_horizontal": stop_when_horizontal
        }
        self.logger = logging.getLogger("ASRFClassifier")
               
    def fit(self, X, y):
        
        # set numbers of trees to 0
        self.warm_start = False
        self.estimators_ = []
        self.n_estimators = 0
        self.warm_start = True
               
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
        
        # create a function that can efficiently compute the Brier score for a probability distribution
        def get_mse_score(y_pred):
            return np.mean((y_pred - y)**2)
        
        # this is a variable that is being used by the supplier
        self.y_pred_oob = np.zeros(y.shape[0])
        
        def supplier_for_mse(alpha): # alpha not used here
            
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
                y_pred_oob_tree = last_tree.predict(X[unsampled_indices])

                # update forest's prediction
                self.y_pred_oob[unsampled_indices] = (y_pred_oob_tree + t * self.y_pred_oob[unsampled_indices]) / (t + 1) # this will converge according to the law of large numbers

            
            # compute performance diffs of relevant other anchors to new one
            return [get_mse_score(self.y_pred_oob)]
        
        # always use Brier score supplier
        grower = ForestGrower(supplier_for_mse, 1, logger = self.logger, random_state = self.random_state, **self.args)
        self.logger.info(f"Starting to grow forest.")
        grower.grow()
        self.histories = grower.histories
        self.logger.info("Forest ready. Histories stored. Leaving function.")